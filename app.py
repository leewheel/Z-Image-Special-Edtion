import os
import time  # 【新增】引入time模块用于退出延迟

# 必须处于文件最顶端：环境配置
os.environ["DIFFUSERS_USE_PEFT_BACKEND"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import torch
import psutil
import random
import re
import uuid
import gc
from datetime import datetime
from PIL import Image, ImageFilter, ImageOps, ImageEnhance

# 配置基础路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 目录配置
DEFAULT_MODEL_PATH = os.path.join(current_dir, "ckpts", "Z-Image-Turbo")
LORA_ROOT = os.path.join(current_dir, "lora")
OUTPUT_ROOT = os.path.join(current_dir, "outputs")
MOD_VAE_DIR = os.path.join(current_dir, "Mod", "vae")
MOD_TRANS_DIR = os.path.join(current_dir, "Mod", "transformer")
for p in [LORA_ROOT, OUTPUT_ROOT, MOD_VAE_DIR, MOD_TRANS_DIR]:
    os.makedirs(p, exist_ok=True)

try:
    import gradio as gr
    from diffusers import ZImagePipeline, ZImageImg2ImgPipeline, AutoencoderKL
    from safetensors.torch import load_file
except ImportError as e:
    print(f"❌ 核心库导入失败: {e}")
    sys.exit(1)

# ==========================================
# 设备探测与硬件报告
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
is_interrupted = False

print("\n" + "="*50)
if DEVICE == "cuda":
    GPU_NAME = torch.cuda.get_device_name(0)
    TOTAL_VRAM = torch.cuda.get_device_properties(0).total_memory
    print(f"✅ 运行模式: [ GPU ]")
    print(f"核心型号: {GPU_NAME}")
    print(f"显存总量: {TOTAL_VRAM/1024**3:.2f} GB")
else:
    TOTAL_VRAM = 0
    print(f"⚠️ 运行模式: [ CPU ]")
print("="*50 + "\n")

# ==========================================
# 显存与工具函数
# ==========================================
def get_vram_info():
    if DEVICE == "cuda":
        reserved = torch.cuda.memory_reserved(0)
        allocated = torch.cuda.memory_allocated(0)
        usage_pct = (reserved / TOTAL_VRAM) * 100 if TOTAL_VRAM > 0 else 0
        vram_str = (
            f"显存占用: {usage_pct:.1f}% "
            f"({reserved/1024**3:.2f}GB / {TOTAL_VRAM/1024**3:.2f}GB)"
        )
    else:
        usage_pct = 0
        vram_str = "显存占用: CPU 模式"

    mem = psutil.virtual_memory()
    ram_str = (
        f"内存占用: {mem.percent:.1f}% "
        f"({(mem.total - mem.available)/1024**3:.2f}GB / {mem.total/1024**3:.2f}GB)"
    )
    status = f"{vram_str} ｜ {ram_str}"
    return usage_pct, status


def auto_flush_vram(threshold=90):
    usage_pct, _ = get_vram_info()
    if usage_pct > threshold:
        gc.collect()
        torch.cuda.empty_cache()
        return True
    return False

def scan_lora_files():
    if not os.path.exists(LORA_ROOT): return []
    return sorted([f for f in os.listdir(LORA_ROOT) if f.lower().endswith(".safetensors")])

def scan_model_items(base_path):
    if not os.path.exists(base_path): return []
    items = []
    for f in os.listdir(base_path):
        full_path = os.path.join(base_path, f)
        if os.path.isdir(full_path):
            items.append(f)
        elif f.lower().endswith((".safetensors", ".bin", ".pt")):
            items.append(f)
    return sorted(items)

# ==========================================
# 全局 LoRA 文件列表 (启动时扫描)
# ==========================================
LORA_FILES = scan_lora_files()
print(f"🔍 已检测到 {len(LORA_FILES)} 个 LoRA 文件，正在生成独立控件...")
if len(LORA_FILES) > 30:
    print("⚠️ 警告: LoRA 数量较多，生成界面可能需要几秒钟...")

# ==========================================
# 模型管理器 (修改版：支持独立权重)
# ==========================================
class ModelManager:
    def __init__(self):
        self.pipe = None 
        self.current_state = {
            "mode": None,      
            "t_choice": None,  
            "v_choice": None,  
        }
        self.current_loras = []
        self.current_weights_map = {} 

    def _clear_pipeline(self):
        if self.pipe is not None:
            print(f"🧹 正在销毁旧管道以释放显存...")
            try:
                self.pipe.unload_lora_weights()
            except:
                pass
            del self.pipe
            self.pipe = None
        if hasattr(sys, 'last_traceback'):
            del sys.last_traceback
        for _ in range(3):
            gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if DEVICE == "cuda":
            res = torch.cuda.memory_reserved(0) / 1024**3
            print(f"✨ 显存已深度清理，当前占用: {res:.2f} GB")

    def _init_pipeline_base(self, mode):
        if mode == 'txt':
            print("🚀 初始化基础 Pipeline (文生图)...")
            return ZImagePipeline.from_pretrained(DEFAULT_MODEL_PATH, torch_dtype=DTYPE, local_files_only=True)
        else:
            print("🚀 初始化基础 Pipeline (图生图)...")
            return ZImageImg2ImgPipeline.from_pretrained(DEFAULT_MODEL_PATH, torch_dtype=DTYPE, local_files_only=True)

    def _inject_components(self, pipe, t_choice, v_choice):
        if t_choice != "default":
            t_path = os.path.join(MOD_TRANS_DIR, t_choice)
            if os.path.isfile(t_path):
                print(f"📦 载入 Transformer: {t_choice}")
                state_dict = load_file(t_path, device="cpu")
                processed = {}
                prefix = "model.diffusion_model."
                for k, v in state_dict.items():
                    new_k = k[len(prefix):] if k.startswith(prefix) else k
                    processed[new_k] = v.to(DTYPE)
                pipe.transformer.load_state_dict(processed, strict=False, assign=True)
                del state_dict, processed, v
                gc.collect()

        if v_choice != "default":
            vae_path = os.path.join(MOD_VAE_DIR, v_choice)
            print(f"📦 载入 VAE: {v_choice}")
            if os.path.isfile(vae_path):
                pipe.vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=DTYPE)
            else:
                pipe.vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=DTYPE)
        return pipe

    def _apply_loras(self, pipe, selected_loras, weights_map):
        if self.current_loras == selected_loras and self.current_weights_map == weights_map:
            return

        print("🎸 正在配置 LoRA (独立权重模式)...")
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass

        if not selected_loras:
            self.current_loras = []
            self.current_weights_map = {}
            return

        active_adapters = []
        adapter_weights = []

        for lora_file in selected_loras:
            adapter_name = re.sub(r"[^a-zA-Z0-9_]", "_", os.path.splitext(lora_file)[0])
            weight = weights_map.get(lora_file, 1.0)
            
            try:
                pipe.load_lora_weights(LORA_ROOT, weight_name=lora_file, adapter_name=adapter_name)
                active_adapters.append(adapter_name)
                adapter_weights.append(weight)
            except Exception as e:
                print(f"⚠️ LoRA {lora_file} 加载失败: {e}")

        if active_adapters:
            pipe.set_adapters(active_adapters, adapter_weights=adapter_weights)
        
        self.current_loras = list(selected_loras)
        self.current_weights_map = dict(weights_map)

    def get_pipeline(self, t_choice, v_choice, selected_loras, weights_map, mode='txt'):
        need_rebuild = (
            self.pipe is None or
            self.current_state["mode"] != mode or
            self.current_state["t_choice"] != t_choice or
            self.current_state["v_choice"] != v_choice
        )

        if need_rebuild:
            self._clear_pipeline() 
            try:
                temp_pipe = self._init_pipeline_base(mode)
                temp_pipe = self._inject_components(temp_pipe, t_choice, v_choice)
                
                if DEVICE == "cuda":
                    print("⚡ 启用 GPU 显存分片加载")
                    temp_pipe.enable_sequential_cpu_offload()
                
                self.pipe = temp_pipe
                
                self.current_state = {
                    "mode": mode,
                    "t_choice": t_choice,
                    "v_choice": v_choice
                }
                self.current_loras = [] 
                self.current_weights_map = {}
                
            except Exception as e:
                self._clear_pipeline()
                raise gr.Error(f"模型加载崩溃: {str(e)}\n请检查显存或模型文件。")

        self._apply_loras(self.pipe, selected_loras, weights_map)
        return self.pipe

manager = ModelManager()

# ==========================================
# 进度回调
# ==========================================
def make_progress_callback(progress, total_steps, refresh_interval=2):
    def _callback(pipe, step, timestep, callback_kwargs):
        global is_interrupted
        if is_interrupted:
            raise gr.Error("🛑 任务已手动停止")
        step_idx = step + 1
        frac = step_idx / total_steps
        status_suffix = ""
        if step_idx % refresh_interval == 0 or step_idx == total_steps:
            _, mem_status = get_vram_info()
            status_suffix = f"\n{mem_status}"
        progress(frac, desc=f"Diffusion Step {step_idx}/{total_steps}{status_suffix}")
        return callback_kwargs
    return _callback

# ==========================================
# 核心逻辑 (解析独立控件传入的参数)
# ==========================================
def process_lora_inputs(lora_checks, lora_weights):
    selected = []
    weights_map = {}
    for i, fname in enumerate(LORA_FILES):
        if i < len(lora_checks) and lora_checks[i]:
            selected.append(fname)
            if i < len(lora_weights):
                weights_map[fname] = lora_weights[i]
            else:
                weights_map[fname] = 1.0
    return selected, weights_map

# 【新增】更新 Prompt UI 的辅助函数
def update_prompt_ui_base(prompt, *lora_ui_args):
    """
    lora_ui_args 包含: checks (N个) + weights (N个)
    """
    num_loras = len(LORA_FILES)
    if num_loras == 0:
        return prompt

    checks = lora_ui_args[:num_loras]
    weights = lora_ui_args[num_loras:num_loras*2]

    # 清除旧的 lora 标签
    clean_p = re.sub(r"\s*<lora:[^>]+>", "", prompt or "").strip()
    
    new_tags = []
    for i, fname in enumerate(LORA_FILES):
        if i < len(checks) and checks[i]:
            w = weights[i] if i < len(weights) else 1.0
            name = os.path.splitext(fname)[0]
            alpha_str = f"{w:.2f}".rstrip("0").rstrip(".")
            new_tags.append(f"<lora:{name}:{alpha_str}>")
    
    if new_tags:
        return f"{clean_p} {' '.join(new_tags)}"
    else:
        return clean_p

# 【修复】使用 *args 接收参数，避免 Gradio 传参顺序问题
def run_inference(*args):
    global is_interrupted
    is_interrupted = False
    
    # 解析参数顺序
    # [prompt, checks(N), weights(N), t, v, w, h, steps, cfg, seed, random, batch, vram_th]
    idx = 0
    prompt = args[idx]; idx += 1
    num_loras = len(LORA_FILES)
    lora_checks = args[idx : idx+num_loras]; idx += num_loras
    lora_weights = args[idx : idx+num_loras]; idx += num_loras
    
    t_choice = args[idx]; idx += 1
    v_choice = args[idx]; idx += 1
    w = args[idx]; idx += 1
    h = args[idx]; idx += 1
    steps = args[idx]; idx += 1
    cfg = args[idx]; idx += 1
    seed = args[idx]; idx += 1
    is_random = args[idx]; idx += 1
    batch_size = args[idx]; idx += 1
    vram_threshold = args[idx]; idx += 1

    auto_flush_vram(vram_threshold)
    clean_w = (int(w) // 16) * 16
    clean_h = (int(h) // 16) * 16
    
    selected_loras, weights_map = process_lora_inputs(lora_checks, lora_weights)
    
    # 构建最终 Prompt
    if selected_loras:
        tags = []
        for f in selected_loras:
            w_val = weights_map.get(f, 1.0)
            name = os.path.splitext(f)[0]
            tags.append(f"<lora:{name}:{w_val:.2f}>")
        clean_p = re.sub(r"\s*<lora:[^>]+>", "", prompt or "").strip()
        final_prompt = f"{clean_p} {' '.join(tags)}"
    else:
        final_prompt = prompt

    try:
        pipe = manager.get_pipeline(t_choice, v_choice, selected_loras, weights_map, mode='txt')
    except Exception as e:
        raise gr.Error(f"模型加载失败: {str(e)}")

    if is_random: seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(DEVICE).manual_seed(int(seed))

    date_folder = datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join(OUTPUT_ROOT, date_folder)
    os.makedirs(save_dir, exist_ok=True)

    results_images = []
    progress = gr.Progress()

    try:
        print(f"🔥 任务启动 | 图片分辨率: {clean_w}x{clean_h} | 种子: {seed}")
        step_callback = make_progress_callback(progress, int(steps))

        for i in range(int(batch_size)):
            if is_interrupted: break
            output = pipe(
                prompt=final_prompt,
                width=clean_w,
                height=clean_h,
                num_inference_steps=int(steps),
                guidance_scale=float(cfg),
                generator=generator,
                callback_on_step_end=step_callback
            ).images[0]

            filename = f"{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:4]}.png"
            path = os.path.join(save_dir, filename)
            output.save(path)
            results_images.append(output)
            _, current_status = get_vram_info()
            yield results_images, seed, current_status

    except Exception as e:
        if "任务已手动停止" in str(e):
            print("🛑 任务已停止")
        else:
            import traceback
            traceback.print_exc()
            raise gr.Error(f"生成中断: {str(e)}")
    finally:
        # del pipe
        auto_flush_vram(vram_threshold)

# 【修复】图生图
def run_img2img(*args, progress=gr.Progress()):
    global is_interrupted
    is_interrupted = False
    
    # [input_image, prompt, checks(N), weights(N), ...fixed...]
    idx = 0
    input_image = args[idx]; idx += 1
    prompt = args[idx]; idx += 1
    
    num_loras = len(LORA_FILES)
    lora_checks = args[idx : idx+num_loras]; idx += num_loras
    lora_weights = args[idx : idx+num_loras]; idx += num_loras
    
    t_choice = args[idx]; idx += 1
    v_choice = args[idx]; idx += 1
    output_width = args[idx]; idx += 1
    output_height = args[idx]; idx += 1
    strength = args[idx]; idx += 1
    steps = args[idx]; idx += 1
    cfg = args[idx]; idx += 1
    seed = args[idx]; idx += 1
    is_random = args[idx]; idx += 1
    batch_size = args[idx]; idx += 1
    vram_threshold = args[idx]; idx += 1

    if input_image is None:
        raise gr.Error("❌ 请先上传图片")
        
    auto_flush_vram(vram_threshold)
    selected_loras, weights_map = process_lora_inputs(lora_checks, lora_weights)

    if selected_loras:
        tags = []
        for f in selected_loras:
            w_val = weights_map.get(f, 1.0)
            name = os.path.splitext(f)[0]
            tags.append(f"<lora:{name}:{w_val:.2f}>")
        clean_p = re.sub(r"\s*<lora:[^>]+>", "", prompt or "").strip()
        final_prompt = f"{clean_p} {' '.join(tags)}"
    else:
        final_prompt = prompt

    if output_width == 0 or output_height == 0:
        orig_w, orig_h = input_image.size
        aspect = orig_w / orig_h
        target_size = 1024
        if aspect > 1:
            target_w, target_h = target_size, max(512, int(target_size / aspect))
        else:
            target_h, target_w = target_size, max(512, int(target_size * aspect))
        target_w = (target_w // 16) * 16
        target_h = (target_h // 16) * 16
    else:
        target_w = (int(output_width) // 16) * 16
        target_h = (int(output_height) // 16) * 16

    input_image = input_image.convert("RGB").resize((target_w, target_h))

    if is_random: seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(DEVICE).manual_seed(int(seed))

    date_folder = datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join(OUTPUT_ROOT, date_folder)
    os.makedirs(save_dir, exist_ok=True)

    results = []
    pipe = None
    
    try:
        pipe = manager.get_pipeline(t_choice, v_choice, selected_loras, weights_map, mode='img')

        for i in progress.tqdm(range(int(batch_size)), desc="图生图生成中"):
            if is_interrupted: break
            torch.cuda.ipc_collect()
            step_callback = make_progress_callback(progress, int(steps))

            output = pipe(
                prompt=final_prompt,
                image=input_image,
                strength=float(strength),
                num_inference_steps=int(steps),
                guidance_scale=0.0,
                generator=generator,
                callback_on_step_end=step_callback
            ).images[0]

            filename = f"img2img_{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:4]}.png"
            path = os.path.join(save_dir, filename)
            output.save(path)
            results.append(path)

    except Exception as e:
        if "任务已手动停止" in str(e):
            print("🛑 任务已停止")
        else:
            import traceback
            traceback.print_exc()
            raise gr.Error(f"生成中断: {str(e)}")
    finally:
        del pipe
        auto_flush_vram(vram_threshold)
        _, current_status = get_vram_info()

    return results, seed, current_status

# 【修复】融合图
def run_fusion_img(*args, progress=gr.Progress()):
    global is_interrupted
    is_interrupted = False
    
    # [image1, image2, prompt, checks(N), weights(N), ...fixed...]
    idx = 0
    image1 = args[idx]; idx += 1
    image2 = args[idx]; idx += 1
    prompt = args[idx]; idx += 1
    
    num_loras = len(LORA_FILES)
    lora_checks = args[idx : idx+num_loras]; idx += num_loras
    lora_weights = args[idx : idx+num_loras]; idx += num_loras
    
    t_choice = args[idx]; idx += 1
    v_choice = args[idx]; idx += 1
    output_width = args[idx]; idx += 1
    output_height = args[idx]; idx += 1
    blend_strength = args[idx]; idx += 1
    strength = args[idx]; idx += 1
    steps = args[idx]; idx += 1
    cfg = args[idx]; idx += 1
    seed = args[idx]; idx += 1
    is_random = args[idx]; idx += 1
    batch_size = args[idx]; idx += 1
    vram_threshold = args[idx]; idx += 1

    if image1 is None or image2 is None:
        raise gr.Error("❌ 请上传两张参考图片")
        
    auto_flush_vram(vram_threshold)
    selected_loras, weights_map = process_lora_inputs(lora_checks, lora_weights)

    if selected_loras:
        tags = []
        for f in selected_loras:
            w_val = weights_map.get(f, 1.0)
            name = os.path.splitext(f)[0]
            tags.append(f"<lora:{name}:{w_val:.2f}>")
        clean_p = re.sub(r"\s*<lora:[^>]+>", "", prompt or "").strip()
        final_prompt = f"{clean_p} {' '.join(tags)}"
    else:
        final_prompt = prompt

    if output_width == 0 or output_height == 0:
        orig_w, orig_h = image1.size
        aspect = orig_w / orig_h
        target_size = 1024
        if aspect > 1:
            target_w, target_h = target_size, max(512, int(target_size / aspect))
        else:
            target_h, target_w = target_size, max(512, int(target_size * aspect))
        target_w = (target_w // 16) * 16
        target_h = (target_h // 16) * 16
    else:
        target_w = (int(output_width) // 16) * 16
        target_h = (int(output_height) // 16) * 16

    image1 = image1.convert("RGB").resize((target_w, target_h))
    image2 = image2.convert("RGB").resize((target_w, target_h))
    blended_image = Image.blend(image1, image2, float(blend_strength))

    if is_random: seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(DEVICE).manual_seed(int(seed))

    date_folder = datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join(OUTPUT_ROOT, date_folder)
    os.makedirs(save_dir, exist_ok=True)

    results = []
    pipe = None
    
    try:
        pipe = manager.get_pipeline(t_choice, v_choice, selected_loras, weights_map, mode='img')

        for i in progress.tqdm(range(int(batch_size)), desc="融合生成中"):
            if is_interrupted: break
            torch.cuda.ipc_collect()
            step_callback = make_progress_callback(progress, int(steps))

            output = pipe(
                prompt=final_prompt,
                image=blended_image,
                strength=float(strength),
                num_inference_steps=int(steps),
                guidance_scale=0.0,
                generator=generator,
                callback_on_step_end=step_callback
            ).images[0]

            filename = f"fusion_{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:4]}.png"
            path = os.path.join(save_dir, filename)
            output.save(path)
            results.append(path)

    except Exception as e:
        if "任务已手动停止" in str(e):
            print("🛑 任务已停止")
        else:
            import traceback
            traceback.print_exc()
            raise gr.Error(f"生成中断: {str(e)}")
    finally:
        del pipe
        auto_flush_vram(vram_threshold)
        _, current_status = get_vram_info()

    return results, seed, current_status

# ==========================================
# UI 界面
# ==========================================
# 【新增】定义JS退出脚本：关闭窗口或显示黑屏
js_kill_window = """
function() {
    // 尝试关闭窗口
    setTimeout(function(){ window.close(); }, 1000);
    // 如果无法关闭，则覆盖页面显示提示
    document.body.innerHTML = '<div style="display:flex;justify-content:center;align-items:center;height:100vh;background:#000;color:#fff;font-family:sans-serif;"><h1>🚫 系统已关闭，请直接关闭此标签页</h1></div>';
    document.body.style.backgroundColor = "black";
    document.body.style.overflow = "hidden";
    return [];
}
"""

# JS退出脚本：关闭窗口或显示黑屏
js_kill_window = """
function() {
    // 尝试关闭窗口
    setTimeout(function(){ window.close(); }, 1000);
    // 如果无法关闭，则覆盖页面显示提示
    document.body.innerHTML = '<div style="display:flex;justify-content:center;align-items:center;height:100vh;background:#000;color:#fff;font-family:sans-serif;"><h1>🚫 系统已关闭，请直接关闭此标签页</h1></div>';
    document.body.style.backgroundColor = "black";
    document.body.style.overflow = "hidden";
    return [];
}
"""

# Python退出函数：关闭进程
def kill_system_process():
    print("🛑 正在执行一键退出程序...")
    try:
        # 1. 优先关闭启动器 (Windows)
        os.system("taskkill /F /IM Z-Image-Launcher.exe")
    except Exception:
        pass

    # 延迟1秒，确保前端JS有机会执行
    time.sleep(1)

    try:
        # 2. 强制杀掉所有 Python 进程
        os.system("taskkill /F /IM python.exe")
    except Exception:
        pass

    # 3. 最后自杀（如果上面没杀掉自己的话）
    sys.exit(0)

with gr.Blocks(title="造相 Z-Image Pro Studio | 作者: ") as demo:

    print('\n' + '!'*60)
    print('  本软件由 Leewheel 免费分享，严禁售卖！')
    print('!'*60 + '\n')
    gr.Warning('本软件由 Leewheel 免费分享。如果你是付费购买，你被骗了！', duration=20)
    # 【修改】顶部增加一键退出按钮
    with gr.Row(elem_id="header_row"):
        gr.Markdown("# 🎨 造相 Z-Image Pro Studio | 作者:  Leewheel(V1.00C)")
        exit_btn = gr.Button("❌ 一键退出系统", variant="stop", scale=0, min_width=150)
        
    vram_info_display = gr.Markdown("显存状态加载中...")

    with gr.Tabs():
        # --- 文成图 ---
        with gr.Tab("文成图"):
            with gr.Row():
                with gr.Column(scale=4):
                    prompt_input = gr.Textbox(label="Prompt", lines=4)
                    manual_flush_btn = gr.Button("🧹 清理显存", size="sm", variant="secondary")
                    vram_threshold_slider = gr.Slider(50, 98, 90, step=1, label="自动清理阈值 (%)")
                    
                    # 【核心修改】动态生成每个 LoRA 的控件
                    with gr.Accordion("LoRA 权重设置 (每个 LoRA 独立调节)", open=False):
                        txt_lora_checks = []
                        txt_lora_sliders = []
                        
                        if not LORA_FILES:
                            gr.Markdown("*未检测到 LoRA 文件*")
                        else:
                            for fname in LORA_FILES:
                                with gr.Row():
                                    # 复选框
                                    chk = gr.Checkbox(label=fname, value=False, scale=1, container=False)
                                    # 滑块
                                    sld = gr.Slider(0, 2.0, 1.0, step=0.05, label="权重", scale=4)
                                    txt_lora_checks.append(chk)
                                    txt_lora_sliders.append(sld)

                    with gr.Accordion("模型设置", open=True):
                        refresh_models_btn = gr.Button("🔄 刷新底模/VAE", size="sm")
                        t_drop = gr.Dropdown(label="Transformer", choices=["default"] + scan_model_items(MOD_TRANS_DIR), value="default")
                        v_drop = gr.Dropdown(label="VAE", choices=["default"] + scan_model_items(MOD_VAE_DIR), value="default")
                        with gr.Row():
                            width_s = gr.Slider(512, 2048, 1024, step=16, label="宽 (16倍数)")
                            height_s = gr.Slider(512, 2048, 1024, step=16, label="高 (16倍数)")
                        step_s = gr.Slider(1, 50, 8, label="步数")
                        cfg_s = gr.Slider(0, 10, 0, label="CFG")
                        batch_s = gr.Slider(1, 32, 1, step=1, label="生成张数")
                        seed_n = gr.Number(label="种子", value=42, precision=0)
                        random_c = gr.Checkbox(label="随机种子", value=True)

                    with gr.Row():
                        run_btn = gr.Button("🚀 开始生成", variant="primary", size="lg")
                        stop_btn = gr.Button("🛑 停止生成", variant="stop", size="lg", interactive=False)

                with gr.Column(scale=6):
                    res_gallery = gr.Gallery(label="输出结果", columns=2, height="80vh")
                    res_seed = gr.Number(label="种子", interactive=False)
                    vram_info_display = gr.Markdown("显存状态加载中...")

        # --- 图片编辑 ---
        with gr.Tab("图片编辑"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(label="上传图片", type="pil")
                    with gr.Group():
                        rotate_angle = gr.Slider(-360, 360, 0, step=1, label="旋转角度 (度)")
                        crop_x = gr.Slider(0, 100, 0, step=1, label="裁剪 X (%)")
                        crop_y = gr.Slider(0, 100, 0, step=1, label="裁剪 Y (%)")
                        crop_width = gr.Slider(0, 100, 100, step=1, label="裁剪宽度 (%)")
                        crop_height = gr.Slider(0, 100, 100, step=1, label="裁剪高度 (%)")
                        flip_horizontal = gr.Checkbox(label="水平翻转")
                        flip_vertical = gr.Checkbox(label="垂直翻转")
                    edit_btn = gr.Button("开始编辑", variant="primary")
                with gr.Column():
                    edited_image_output = gr.Image(label="编辑后的图片", type="pil")
                    with gr.Group():
                        apply_filter = gr.Dropdown(["模糊", "轮廓", "细节", "边缘增强", "更多边缘增强", "浮雕", "查找边缘", "锐化", "平滑", "更多平滑"], label="应用滤镜")
                        brightness = gr.Slider(-100, 100, 0, step=1, label="亮度调整 (%)")
                        contrast = gr.Slider(-100, 100, 0, step=1, label="对比度调整 (%)")
                        saturation = gr.Slider(-100, 100, 0, step=1, label="饱和度调整 (%)")

            def edit_image(image, angle, x, y, width, height, hflip, vflip, filter, brightness, contrast, saturation):
                if image is None: return None
                if angle != 0: image = image.rotate(angle, expand=True)
                if x or y or width < 100 or height < 100:
                    original_width, original_height = image.size
                    left = int(original_width * x / 100)
                    top = int(original_height * y / 100)
                    right = int(original_width * (x + width) / 100)
                    bottom = int(original_height * (y + height) / 100)
                    image = image.crop((left, top, right, bottom))
                if hflip: image = ImageOps.mirror(image)
                if vflip: image = ImageOps.flip(image)
                if filter:
                    filter_map = {
                        "模糊": ImageFilter.BLUR, "轮廓": ImageFilter.CONTOUR, "细节": ImageFilter.DETAIL,
                        "边缘增强": ImageFilter.EDGE_ENHANCE, "更多边缘增强": ImageFilter.EDGE_ENHANCE_MORE,
                        "浮雕": ImageFilter.EMBOSS, "查找边缘": ImageFilter.FIND_EDGES,
                        "锐化": ImageFilter.SHARPEN, "平滑": ImageFilter.SMOOTH, "更多平滑": ImageFilter.SMOOTH_MORE
                    }
                    filter_func = filter_map.get(filter)
                    if filter_func: image = image.filter(filter_func)
                if brightness != 0:
                    enhancer = ImageEnhance.Brightness(image)
                    image = enhancer.enhance(1 + brightness / 100)
                if contrast != 0:
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(1 + contrast / 100)
                if saturation != 0:
                    enhancer = ImageEnhance.Color(image)
                    image = enhancer.enhance(1 + saturation / 100)
                return image

            edit_btn.click(
                fn=edit_image,
                inputs=[image_input, rotate_angle, crop_x, crop_y, crop_width, crop_height, flip_horizontal, flip_vertical, apply_filter, brightness, contrast, saturation],
                outputs=edited_image_output
            )

        # --- 图生图 ---
        with gr.Tab("图生图"):
            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Group():
                        img2img_input = gr.Image(label="上传参考图", type="pil")
                        img2img_prompt = gr.Textbox(label="Prompt (推荐)", lines=2, placeholder="描述你想要生成的画面...")
                        img2img_flush = gr.Button("🧹 清理显存", size="sm", variant="secondary")
                        
                        # 【核心修改】图生图独立控件
                        with gr.Accordion("LoRA 权重设置 (独立调节)", open=False):
                            i2i_lora_checks = []
                            i2i_lora_sliders = []
                            if not LORA_FILES:
                                gr.Markdown("*未检测到 LoRA 文件*")
                            else:
                                for fname in LORA_FILES:
                                    with gr.Row():
                                        chk = gr.Checkbox(label=fname, value=False, scale=1, container=False)
                                        sld = gr.Slider(0, 2.0, 1.0, step=0.05, label="权重", scale=4)
                                        i2i_lora_checks.append(chk)
                                        i2i_lora_sliders.append(sld)

                    with gr.Accordion("模型与参数", open=True):
                        img2img_refresh_models = gr.Button("🔄 刷新底模/VAE", size="sm")
                        img2img_t_drop = gr.Dropdown(label="Transformer", choices=["default"] + scan_model_items(MOD_TRANS_DIR), value="default")
                        img2img_v_drop = gr.Dropdown(label="VAE", choices=["default"] + scan_model_items(MOD_VAE_DIR), value="default")
                        with gr.Row():
                            img2img_width_s = gr.Slider(0, 2048, 0, step=16, label="输出宽 (0=自动保持比例)")
                            img2img_height_s = gr.Slider(0, 2048, 0, step=16, label="输出高 (0=自动保持比例)")
                        gr.Markdown("**提示：** 宽高都为0时自动保持上传图比例并接近1024；手动设置大于512时生效")
                        img2img_strength = gr.Slider(0.0, 1.0, 0.75, step=0.01, label="重绘强度")
                        img2img_steps = gr.Slider(1, 100, 12, step=1, label="步数")
                        img2img_cfg = gr.Number(value=0.0, label="CFG（Turbo模型固定为0.0）", interactive=False)
                        img2img_batch = gr.Slider(1, 8, 1, step=1, label="张数")
                        img2img_seed = gr.Number(label="种子", value=42, precision=0)
                        img2img_random = gr.Checkbox(label="随机种子", value=True)
                    with gr.Row():
                        img2img_run_btn = gr.Button("🚀 生成", variant="primary", size="lg")
                        img2img_stop_btn = gr.Button("🛑 停止", variant="stop", size="lg", interactive=False)
                with gr.Column(scale=6):
                    img2img_gallery = gr.Gallery(label="图生图结果", columns=2, height="80vh")
                    img2img_res_seed = gr.Number(label="种子", interactive=False)

        # --- 融合图 ---
        with gr.Tab("融合图"):
            gr.Markdown("**融合2张图片**：图片1提供主要结构/姿势，图片2提供细节/脸部/风格。")
            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Group():
                        fusion_input1 = gr.Image(label="图片1（主结构/姿势）", type="pil")
                        fusion_input2 = gr.Image(label="图片2（细节/脸部/风格）", type="pil")
                        fusion_prompt = gr.Textbox(label="融合描述 Prompt", lines=3)
                        fusion_flush = gr.Button("🧹 清理显存", size="sm", variant="secondary")
                        
                        # 【核心修改】融合图独立控件
                        with gr.Accordion("LoRA 权重设置 (独立调节)", open=False):
                            fusion_lora_checks = []
                            fusion_lora_sliders = []
                            if not LORA_FILES:
                                gr.Markdown("*未检测到 LoRA 文件*")
                            else:
                                for fname in LORA_FILES:
                                    with gr.Row():
                                        chk = gr.Checkbox(label=fname, value=False, scale=1, container=False)
                                        sld = gr.Slider(0, 2.0, 1.0, step=0.05, label="权重", scale=4)
                                        fusion_lora_checks.append(chk)
                                        fusion_lora_sliders.append(sld)

                    with gr.Accordion("模型与参数", open=True):
                        fusion_refresh_models = gr.Button("🔄 刷新底模/VAE", size="sm")
                        fusion_t_drop = gr.Dropdown(label="Transformer", choices=["default"] + scan_model_items(MOD_TRANS_DIR), value="default")
                        fusion_v_drop = gr.Dropdown(label="VAE", choices=["default"] + scan_model_items(MOD_VAE_DIR), value="default")
                        with gr.Row():
                            fusion_width_s = gr.Slider(0, 2048, 0, step=16, label="输出宽 (0=自动保持比例)")
                            fusion_height_s = gr.Slider(0, 2048, 0, step=16, label="输出高 (0=自动保持比例)")
                        gr.Markdown("**提示：** 宽高都为0时自动保持图片1比例并接近1024")
                        with gr.Row():
                            fusion_blend = gr.Slider(0.0, 1.0, 0.5, step=0.05, label="图片2混合强度 (0=全用图片1, 1=全用图片2)")
                            fusion_strength = gr.Slider(0.0, 1.0, 0.7, step=0.05, label="重绘强度 (越高变化越大)")
                        fusion_steps = gr.Slider(1, 100, 15, step=1, label="步数")
                        fusion_cfg = gr.Number(value=0.0, label="CFG（固定为0.0）", interactive=False)
                        fusion_batch = gr.Slider(1, 8, 1, step=1, label="张数")
                        fusion_seed = gr.Number(label="种子", value=42, precision=0)
                        fusion_random = gr.Checkbox(label="随机种子", value=True)
                    with gr.Row():
                        fusion_run_btn = gr.Button("🚀 开始融合", variant="primary", size="lg")
                        fusion_stop_btn = gr.Button("🛑 停止", variant="stop", size="lg", interactive=False)
                with gr.Column(scale=6):
                    fusion_gallery = gr.Gallery(label="融合结果", columns=2, height="80vh")
                    fusion_res_seed = gr.Number(label="种子", interactive=False)

    # -----------------------
    # UI状态函数
    # -----------------------
    def ui_to_running():
        return gr.update(interactive=False), gr.update(interactive=True)

    def ui_to_idle():
        return gr.update(interactive=True), gr.update(interactive=False)

    def trigger_interrupt():
        global is_interrupted
        is_interrupted = True
        return "🛑 正在强制中断..."

    # -----------------------
    # 按钮事件绑定
    # -----------------------
    
    # 【新增】退出按钮绑定事件
    # 1. 先触发 _js 执行前端清理（变黑、尝试关闭窗口）
    # 2. 然后触发 fn 执行后端杀进程
    # 合并逻辑：一次点击同时触发 JS 和 Python
    exit_btn.click(
        fn=kill_system_process,   # 后端：杀进程
        js=js_kill_window         # 前端：关网页或显示黑屏
    )

    # 文生图
    refresh_models_btn.click(
        fn=lambda: (
            gr.update(choices=["default"] + scan_model_items(MOD_TRANS_DIR)),
            gr.update(choices=["default"] + scan_model_items(MOD_VAE_DIR))
        ),
        outputs=[t_drop, v_drop]
    )
    manual_flush_btn.click(
        fn=lambda: (gc.collect(), torch.cuda.empty_cache(), get_vram_info()[1])[2],
        outputs=vram_info_display
    )

    # 【新增】绑定文生图 Prompt 自动更新
    txt_ui_inputs = [prompt_input] + txt_lora_checks + txt_lora_sliders
    for c in txt_lora_checks + txt_lora_sliders:
        c.change(fn=update_prompt_ui_base, inputs=txt_ui_inputs, outputs=prompt_input)

    inference_event = run_btn.click(
        fn=ui_to_running, 
        outputs=[run_btn, stop_btn]
    ).then(
        fn=run_inference,
        inputs=txt_ui_inputs + [t_drop, v_drop, width_s, height_s, step_s, cfg_s, seed_n, random_c, batch_s, vram_threshold_slider],
        outputs=[res_gallery, res_seed, vram_info_display]
    ).then(
        fn=ui_to_idle,
        outputs=[run_btn, stop_btn]
    )

    stop_btn.click(
        fn=trigger_interrupt,
        outputs=vram_info_display
    ).then(
        fn=ui_to_idle,
        outputs=[run_btn, stop_btn],
        cancels=[inference_event]
    )
    
    # 图生图
    def refresh_all_models_img():
        return gr.update(choices=["default"] + scan_model_items(MOD_TRANS_DIR)), gr.update(choices=["default"] + scan_model_items(MOD_VAE_DIR))
    img2img_refresh_models.click(fn=refresh_all_models_img, outputs=[img2img_t_drop, img2img_v_drop])
    img2img_flush.click(fn=lambda: (gc.collect(), torch.cuda.empty_cache(), get_vram_info()[1])[2], outputs=vram_info_display)

    # 【新增】绑定图生图 Prompt 自动更新
    i2i_ui_inputs = [img2img_prompt] + i2i_lora_checks + i2i_lora_sliders
    for c in i2i_lora_checks + i2i_lora_sliders:
        c.change(fn=update_prompt_ui_base, inputs=i2i_ui_inputs, outputs=img2img_prompt)

    img2img_event = img2img_run_btn.click(fn=ui_to_running, outputs=[img2img_run_btn, img2img_stop_btn])\
        .then(fn=run_img2img,
              inputs=[img2img_input, img2img_prompt] + i2i_lora_checks + i2i_lora_sliders + 
                      [img2img_t_drop, img2img_v_drop, img2img_width_s, img2img_height_s,
                       img2img_strength, img2img_steps, img2img_cfg, img2img_seed, img2img_random, img2img_batch, vram_threshold_slider],
              outputs=[img2img_gallery, img2img_res_seed, vram_info_display])\
        .then(fn=ui_to_idle, outputs=[img2img_run_btn, img2img_stop_btn])

    img2img_stop_btn.click(fn=trigger_interrupt, outputs=vram_info_display).then(fn=ui_to_idle, outputs=[img2img_run_btn, img2img_stop_btn], cancels=[img2img_event])

    # 融合图
    fusion_refresh_models.click(fn=refresh_all_models_img, outputs=[fusion_t_drop, fusion_v_drop])
    fusion_flush.click(fn=lambda: (gc.collect(), torch.cuda.empty_cache(), get_vram_info()[1])[2], outputs=vram_info_display)

    # 【新增】绑定融合图 Prompt 自动更新
    fusion_ui_inputs = [fusion_prompt] + fusion_lora_checks + fusion_lora_sliders
    for c in fusion_lora_checks + fusion_lora_sliders:
        c.change(fn=update_prompt_ui_base, inputs=fusion_ui_inputs, outputs=fusion_prompt)

    fusion_event = fusion_run_btn.click(fn=ui_to_running, outputs=[fusion_run_btn, fusion_stop_btn])\
        .then(fn=run_fusion_img,
              inputs=[fusion_input1, fusion_input2, fusion_prompt] + fusion_lora_checks + fusion_lora_sliders + 
                      [fusion_t_drop, fusion_v_drop, fusion_width_s, fusion_height_s,
                       fusion_blend, fusion_strength, fusion_steps, fusion_cfg, 
                       fusion_seed, fusion_random, fusion_batch, vram_threshold_slider],
              outputs=[fusion_gallery, fusion_res_seed, vram_info_display])\
        .then(fn=ui_to_idle, outputs=[fusion_run_btn, fusion_stop_btn])

    fusion_stop_btn.click(fn=trigger_interrupt, outputs=vram_info_display).then(fn=ui_to_idle, outputs=[fusion_run_btn, fusion_stop_btn], cancels=[fusion_event])

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)