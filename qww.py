import threading
import time
import os
import random
import traceback

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import customtkinter as ctk
    CTK_AVAILABLE = True
except Exception:
    CTK_AVAILABLE = False

from PIL import Image, ImageTk
import cv2
import numpy as np
import pywt
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_nl_means, estimate_sigma

# --- Utility functions ---
def ensure_uint8(img):
    """Convert image array to uint8 safely."""
    if img is None:
        return None
    img = np.asarray(img)
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    else:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def ensure_three_channels(img):
    """Return a 3-channel BGR uint8 image or None if input is None."""
    if img is None:
        return None
    img = ensure_uint8(img)
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    if img.ndim == 3 and img.shape[2] >= 3:
        return img[:, :, :3]
    raise ValueError("Unsupported image shape")

def resize_to_match(img1, img2):
    if img1 is None or img2 is None:
        return img2
    h, w = img1.shape[:2]
    try:
        return cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)
    except Exception:
        return img2

def highlight_changes(original, denoised, threshold=30):
    """
    Overlay red highlights on original where absolute diff between original and denoised
    exceeds threshold. Returns a 3-channel RGB (uint8) image suitable for display.
    """
    if original is None or denoised is None:
        return None
    orig = ensure_three_channels(original)
    den = ensure_three_channels(denoised)
    if orig is None or den is None:
        return None
    if orig.shape != den.shape:
        den = resize_to_match(orig, den)
    try:
        diff = cv2.absdiff(orig, den)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if diff.ndim == 3 else diff
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        overlay = orig.copy()
        red_layer = np.zeros_like(overlay)
        red_layer[:, :, 2] = 255  # red in BGR
        alpha = 0.6
        mask_bool = mask.astype(bool)
        overlay[mask_bool] = cv2.addWeighted(orig[mask_bool].astype(np.float32), 1 - alpha,
                                             red_layer[mask_bool].astype(np.float32), alpha, 0).astype(np.uint8)
        return overlay
    except Exception:
        return orig

# --- Traditional Denoising Algorithms ---

def mean_filter(img):
    img = ensure_uint8(img)
    return cv2.blur(img, (9,9))

def gaussian_filter(img):
    img = ensure_uint8(img)
    return cv2.GaussianBlur(img, (9,9), 3.0)

def median_filter(img):
    img = ensure_uint8(img)
    if len(img.shape)==3 and img.shape[2]==3:
        channels = cv2.split(img)
        return cv2.merge([cv2.medianBlur(c,7) for c in channels])
    return cv2.medianBlur(img,7)

def bilateral_filter(img):
    img = ensure_uint8(img)
    params = (13,120,120)
    if len(img.shape)==3 and img.shape[2]==3:
        return cv2.bilateralFilter(img, *params)
    elif len(img.shape)==2:
        return cv2.bilateralFilter(img, *params)
    else:
        raise ValueError("Input must be grayscale or BGR")

def wavelet_denoise(img):
    img = ensure_uint8(img)
    if len(img.shape) == 3 and img.shape[2] == 3:
        channels = cv2.split(img)
        outs = []
        for c in channels:
            outs.append(wavelet_denoise_single(c))
        return cv2.merge(outs)
    return wavelet_denoise_single(img)

def wavelet_denoise_single(img):
    arr = img.astype(np.float64)
    coeffs = pywt.wavedec2(arr, 'db1', level=2)
    sigma = np.median(np.abs(coeffs[-1][0])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(arr.size)) * 0.12
    coeffs_filt = list(coeffs)
    for i in range(1, len(coeffs)):
        if isinstance(coeffs[i], tuple):
            coeffs_filt[i] = tuple([pywt.threshold(s, threshold, mode='soft') for s in coeffs[i]])
        else:
            coeffs_filt[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
    try:
        rec = pywt.waverec2(coeffs_filt, 'db1')
        rec = np.nan_to_num(rec)
        rec = np.clip(rec, 0, 255)
        if rec.shape != img.shape:
            rec = cv2.resize(rec, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        return rec.astype(np.uint8)
    except Exception:
        return img

# --- WORKING GA Optimization (from your code) ---
def ga_denoise(img, population_size=20, generations=25, callback=None):
    """
    Genetic Algorithm to optimize bilateral filter parameters for denoising.
    Returns denoised image (uint8) or a safe fallback if something fails.
    This function is defensive against None and invalid gene values.
    """
    try:
        img = ensure_uint8(img)
        if img is None:
            raise ValueError("Input image is None")
        if img.ndim == 2:
            cv_img = img
        else:
            cv_img = img.copy()

        def create_individual():
            return [
                random.choice([5, 7, 9, 11, 13, 15]),
                random.randint(50, 200),
                random.randint(50, 200)
            ]

        def safe_params_of(ind):
            defaults = [9, 75, 75]
            if ind is None:
                return defaults[:]
            out = []
            for i, val in enumerate(ind[:3]):
                try:
                    if val is None:
                        out.append(defaults[i])
                    else:
                        ival = int(val)
                        out.append(ival)
                except Exception:
                    out.append(defaults[i])
            d, sc, ss = out
            if d < 3 or d > 15:
                d = 9
            if d % 2 == 0:
                d = d + 1 if d < 15 else 15
            sc = max(1, min(255, sc))
            ss = max(1, min(255, ss))
            return [d, sc, ss]

        def safe_fitness(d, sc, ss):
            try:
                d, sc, ss = safe_params_of([d, sc, ss])
                den = cv2.bilateralFilter(cv_img, d, sc, ss)
                channel_axis = -1 if (den.ndim == 3 and den.shape[2] == 3) else None
                score = ssim(cv_img, den, channel_axis=channel_axis, data_range=255)
                if score is None or np.isnan(score) or np.isinf(score):
                    return 0.0
                return float(score)
            except Exception:
                return 0.0

        population = [create_individual() for _ in range(population_size)]
        best_individual = population[0][:]
        best_fitness = -1.0

        for gen in range(generations):
            fitness_list = []
            for ind in population:
                try:
                    fit = safe_fitness(ind[0], ind[1], ind[2])
                except Exception:
                    fit = 0.0
                fitness_list.append(fit)
                if fit > best_fitness:
                    best_fitness = fit
                    best_individual = ind[:]

            if callback:
                try:
                    safe_params = safe_params_of(best_individual)
                    fval = float(best_fitness) if best_fitness is not None else 0.0
                    callback(gen + 1, fval, safe_params)
                except Exception:
                    try:
                        callback(gen + 1, 0.0, [9, 75, 75])
                    except Exception:
                        pass

            try:
                sorted_indices = sorted(range(len(fitness_list)), key=lambda i: fitness_list[i], reverse=True)
                new_population = []
                elite_count = max(1, population_size // 4)
                for i in sorted_indices[:elite_count]:
                    new_population.append(population[i][:])

                while len(new_population) < population_size:
                    parent_pool_size = max(2, population_size // 2)
                    p1_idx = sorted_indices[random.randint(0, parent_pool_size - 1)]
                    p2_idx = sorted_indices[random.randint(0, parent_pool_size - 1)]
                    parent1 = population[p1_idx]
                    parent2 = population[p2_idx]

                    child = [
                        parent1[0] if random.random() < 0.5 else parent2[0],
                        parent1[1] if random.random() < 0.5 else parent2[1],
                        parent1[2] if random.random() < 0.5 else parent2[2]
                    ]

                    if random.random() < 0.2:
                        gene = random.randint(0, 2)
                        if gene == 0:
                            child[0] = random.choice([5, 7, 9, 11, 13, 15])
                        elif gene == 1:
                            child[1] = max(30, min(200, int(child[1]) + random.randint(-30, 30)))
                        else:
                            child[2] = max(30, min(200, int(child[2]) + random.randint(-30, 30)))

                    new_population.append(child)
                population = new_population
            except Exception:
                population = [create_individual() for _ in range(population_size)]

        d, sc, ss = safe_params_of(best_individual)
        if d < 3:
            d = 9
        if d % 2 == 0:
            d += 1
        d = min(max(d, 3), 15)
        sc = min(max(sc, 30), 200)
        ss = min(max(ss, 30), 200)

        denoised = cv2.bilateralFilter(cv_img, d, sc, ss)
        return denoised

    except Exception as e:
        print("GA failed completely:", e)
        traceback.print_exc()
        try:
            return cv2.bilateralFilter(ensure_uint8(img), 9, 75, 75)
        except Exception:
            return ensure_uint8(img)

def fuzzy_denoise(img):
    img = ensure_uint8(img)
    
    def fuzzy_membership(diff, low=10, high=50):
        if diff < low: return 0.0
        elif diff > high: return 1.0
        else: return (diff - low) / (high - low)
    
    def process_channel(channel):
        output = channel.copy().astype(np.float32)
        h, w = channel.shape
        pad = 2
        padded = cv2.copyMakeBorder(channel, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        for i in range(pad, h + pad):
            for j in range(pad, w + pad):
                center = float(padded[i, j])
                window = padded[i-pad:i+pad+1, j-pad:j+pad+1].flatten()
                local_mean = np.mean(window)
                local_std = np.std(window)
                diff = abs(center - local_mean)
                noise_membership = fuzzy_membership(diff, low=local_std*0.5, high=local_std*2.0)
                if noise_membership > 0.5:
                    output[i-pad, j-pad] = np.median(window)
                elif noise_membership > 0.2:
                    median_val = np.median(window)
                    output[i-pad, j-pad] = center * (1 - noise_membership) + median_val * noise_membership
        return np.clip(output, 0, 255).astype(np.uint8)
    
    if len(img.shape) == 3 and img.shape[2] == 3:
        channels = cv2.split(img)
        return cv2.merge([process_channel(c) for c in channels])
    else:
        return process_channel(img)

def hybrid_denoise(img):
    img = ensure_uint8(img)
    if len(img.shape)==3 and img.shape[2]==3:
        sigma_est = np.mean(estimate_sigma(img, channel_axis=-1))
        nlm_out = denoise_nl_means(img, h=1.15*sigma_est, fast_mode=True, patch_size=5, patch_distance=11, channel_axis=-1)
        nlm_out = ensure_uint8(nlm_out)
    else:
        sigma_est = estimate_sigma(img)
        nlm_out = denoise_nl_means(img, h=1.15*sigma_est, fast_mode=True, patch_size=5, patch_distance=11)
        nlm_out = ensure_uint8(nlm_out)
    return wavelet_denoise(nlm_out)

def pso_denoise(img, population_size=15, generations=20):
    img = ensure_uint8(img)
    pop = [[random.randint(7,19),random.randint(60,200),random.randint(60,200)] for _ in range(population_size)]
    best = pop[0]
    best_score = -1.0
    channel_axis = -1 if len(img.shape)==3 and img.shape[2]==3 else None
    for generation in range(generations):
        for p in pop:
            try:
                d = int(p[0]); p1 = int(p[1]); p2 = int(p[2])
                if d < 3: d = 3
                if d % 2 == 0: d += 1
                den = cv2.bilateralFilter(img, d, p1, p2)
                score = ssim(img, den, channel_axis=channel_axis)
                if np.isnan(score) or np.isinf(score):
                    score = -1.0
            except Exception:
                score = -1.0
            if score > best_score:
                best_score = score
                best = [d, p1, p2]
        for i in range(len(pop)):
            for j in range(3):
                pop[i][j] = int(pop[i][j] + random.choice([-1,0,1]) * (best[j] - pop[i][j]) * 0.5)
    try:
        d, p1, p2 = [int(x) for x in best]
        if d < 3: d = 3
        if d % 2 == 0: d += 1
        return cv2.bilateralFilter(img, d, p1, p2)
    except Exception:
        return cv2.bilateralFilter(img, 7, 50, 50)

ALL_METHODS = [
    ("Mean Filter", lambda img: mean_filter(img)),
    ("Gaussian Filter", lambda img: gaussian_filter(img)),
    ("Median Filter", lambda img: median_filter(img)),
    ("Bilateral Filter", lambda img: bilateral_filter(img)),
    ("Wavelet Denoising", lambda img: wavelet_denoise(img)),
    ("GA Optimization", lambda img: ga_denoise(img)),
    ("Fuzzy Logic", lambda img: fuzzy_denoise(img)),
    ("Hybrid (NLM+Wavelet)", lambda img: hybrid_denoise(img)),
    ("PSO Optimization", lambda img: pso_denoise(img)),
]

# --- Main UI ---

class ImageDenoiserUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Denoiser - Complete View")
        try:
            self.root.state('zoomed')
        except Exception:
            pass
        
        self.orig_img = None
        self.results = []
        self.image_refs = []
        
        if CTK_AVAILABLE:
            ctk.set_appearance_mode("light")
            self._init_ui()
        else:
            self._init_basic_ui()
    
    def _init_ui(self):
        main = ctk.CTkFrame(self.root, fg_color=("#e0e7ff", "#1e1b4b"))
        main.pack(fill="both", expand=True)
        
        top = ctk.CTkFrame(main, fg_color=("#3b82f6", "#1e40af"), height=60, corner_radius=0)
        top.pack(fill="x", padx=0, pady=0)
        top.pack_propagate(False)
        
        top_line = ctk.CTkFrame(top, fg_color=("#60a5fa", "#2563eb"), height=2, corner_radius=0)
        top_line.pack(fill="x", side="bottom")
        
        title_frame = ctk.CTkFrame(top, fg_color="transparent")
        title_frame.pack(fill="both", expand=True, padx=15, pady=8)
        
        ctk.CTkLabel(
            title_frame, 
            text="🎨 IMAGE DENOISER PRO", 
            font=("Arial", 18, "bold"), 
            text_color="white"
        ).pack(side="left")
        
        btn_frame = ctk.CTkFrame(title_frame, fg_color="transparent")
        btn_frame.pack(side="right")
        
        self.load_btn = ctk.CTkButton(
            btn_frame, 
            text="📁 Load", 
            command=self.load_image, 
            font=("Arial", 12, "bold"), 
            fg_color=("#10b981", "#059669"),
            hover_color=("#059669", "#047857"),
            width=110, 
            height=32,
            corner_radius=8
        )
        self.load_btn.pack(side="left", padx=8)
        
        self.process_btn = ctk.CTkButton(
            btn_frame, 
            text="🚀 Process", 
            command=self.process_all, 
            font=("Arial", 12, "bold"), 
            fg_color=("#f59e0b", "#d97706"),
            hover_color=("#d97706", "#b45309"),
            width=110, 
            height=32,
            corner_radius=8
        )
        self.process_btn.pack(side="left", padx=8)
        
        middle = ctk.CTkFrame(main, fg_color=("#f3f4f6", "#111827"), corner_radius=0)
        middle.pack(fill="both", expand=True, padx=0, pady=0)
        
        canvas_container = ctk.CTkFrame(middle, fg_color=("#ffffff", "#1f2937"), corner_radius=0)
        canvas_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.canvas = tk.Canvas(canvas_container, bg=("#ffffff"), highlightthickness=0, bd=0)
        scrollbar = ctk.CTkScrollbar(canvas_container, orientation="horizontal", command=self.canvas.xview)
        
        self.display_frame = tk.Frame(self.canvas, bg=("#ffffff"))
        self.display_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        
        self.canvas.create_window((0, 0), window=self.display_frame, anchor="nw")
        self.canvas.configure(xscrollcommand=scrollbar.set)
        
        self.canvas.pack(side="top", fill="both", expand=True, padx=5, pady=5)
        scrollbar.pack(side="bottom", fill="x", padx=5, pady=(0, 5))
        
        bottom = ctk.CTkFrame(main, fg_color="transparent", height=100)
        bottom.pack(fill="x", padx=15, pady=(0, 15))
        bottom.pack_propagate(False)
        
        best_frame = ctk.CTkFrame(
            bottom, 
            fg_color=("#d1fae5", "#064e3b"), 
            corner_radius=10, 
            border_width=2, 
            border_color=("#10b981", "#059669")
        )
        best_frame.pack(side="left", fill="both", expand=True, padx=(0, 8))
        
        best_header = ctk.CTkFrame(best_frame, fg_color=("#10b981", "#059669"), corner_radius=8, height=30)
        best_header.pack(fill="x", padx=6, pady=6)
        
        ctk.CTkLabel(
            best_header, 
            text="🏆 BEST", 
            font=("Arial", 12, "bold"), 
            text_color="white"
        ).pack(pady=5)
        
        self.best_label = ctk.CTkLabel(
            best_frame, 
            text="Method: -\nSSIM: - | Time: -", 
            font=("Arial", 10, "bold"), 
            text_color=("#065f46", "#6ee7b7"),
            justify="left"
        )
        self.best_label.pack(pady=8)
        
        history_frame = ctk.CTkFrame(
            bottom, 
            fg_color=("#dbeafe", "#1e3a8a"), 
            corner_radius=10, 
            border_width=2, 
            border_color=("#3b82f6", "#1d4ed8")
        )
        history_frame.pack(side="right", fill="both", expand=True, padx=(8, 0))
        
        history_header = ctk.CTkFrame(history_frame, fg_color=("#3b82f6", "#1d4ed8"), corner_radius=8, height=30)
        history_header.pack(fill="x", padx=6, pady=6)
        
        ctk.CTkLabel(
            history_header, 
            text="📝 LOG", 
            font=("Arial", 12, "bold"), 
            text_color="white"
        ).pack(pady=5)
        
        self.history_box = ctk.CTkTextbox(
            history_frame, 
            font=("Consolas", 8), 
            fg_color=("#ffffff", "#1e293b"),
            text_color=("#1e293b", "#cbd5e1"),
            height=40,
            corner_radius=8
        )
        self.history_box.pack(fill="both", expand=True, padx=6, pady=(0, 6))
    
    def _init_basic_ui(self):
        pass
    
    def log(self, text):
        try:
            ts = time.strftime('%H:%M:%S')
            self.history_box.insert("end", f"{ts} - {text}\n")
            self.history_box.see("end")
        except:
            pass
    
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if not path:
            return
        
        try:
            pil = Image.open(path).convert("RGB")
            arr = np.asarray(pil)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            self.orig_img = ensure_uint8(bgr)
            self.results = []
            self._clear_display()
            self.log(f"✓ Loaded: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load: {e}")
            traceback.print_exc()
    
    def process_all(self):
        if self.orig_img is None:
            messagebox.showerror("Error", "Load an image first!")
            return
        
        self.results = []
        self._clear_display()
        threading.Thread(target=self._run_processing, daemon=True).start()
    
    def _run_processing(self):
        self._set_busy(True)
        best_ssim = -1
        best_method = "-"
        best_time = 0
        
        try:
            for name, func in ALL_METHODS:
                self.log(f"⚙️ {name}...")
                start = time.time()
                
                try:
                    denoised = func(self.orig_img)
                    denoised = ensure_uint8(denoised)
                    
                    channel_axis = -1 if self.orig_img.ndim==3 and self.orig_img.shape[2]==3 else None
                    score = ssim(self.orig_img, denoised, channel_axis=channel_axis, data_range=255)
                    if np.isnan(score) or np.isinf(score): score = 0.0
                except Exception as ex:
                    self.log(f"❌ {name}: {ex}")
                    traceback.print_exc()
                    denoised = None
                    score = 0.0
                
                elapsed = time.time() - start
                
                if score > best_ssim:
                    best_ssim = score
                    best_method = name
                    best_time = elapsed
                
                highlighted = highlight_changes(self.orig_img, denoised, threshold=30) if denoised is not None else None
                
                self.results.append({
                    "method": name,
                    "denoised": denoised,
                    "highlighted": highlighted,
                    "ssim": score,
                    "time": elapsed
                })
                
                self.log(f"✓ {name} - SSIM: {score:.4f} - Time: {elapsed:.2f}s")
                self.root.after(0, self._update_display)
            
            self.root.after(0, lambda: self.best_label.configure(
                text=f"Method: {best_method}\nSSIM: {best_ssim:.4f} | Time: {best_time:.2f}s"
            ))
            
        except Exception as ex:
            self.log(f"❌ Failed: {ex}")
            traceback.print_exc()
        finally:
            self._set_busy(False)
    
    def _clear_display(self):
        for widget in self.display_frame.winfo_children():
            widget.destroy()
        self.image_refs = []
    
    def _update_display(self):
        self._clear_display()
        
        self._add_card("Original", self.orig_img, None, "-", 0)
        
        for i, result in enumerate(self.results):
            self._add_card(
                result["method"],
                result["denoised"],
                result["highlighted"],
                result["ssim"],
                result["time"]
            )
    
    def _add_card(self, title, denoised_img, highlighted_img, ssim_score, elapsed):
        col = len([w for w in self.display_frame.winfo_children()])
        
        card = ctk.CTkFrame(
            self.display_frame, 
            fg_color="white", 
            corner_radius=12, 
            border_width=2, 
            border_color=("#e5e7eb", "#374151")
        )
        card.grid(row=0, column=col, padx=10, pady=10, sticky="n")
        
        title_bg = ctk.CTkFrame(card, fg_color=("#f3f4f6", "#1f2937"), corner_radius=8, height=32)
        title_bg.pack(fill="x", padx=6, pady=6)
        
        ctk.CTkLabel(
            title_bg, 
            text=title, 
            font=("Arial", 12, "bold"), 
            text_color=("#1e293b", "#e2e8f0")
        ).pack(pady=6)
        
        if denoised_img is not None:
            tk_img = self._to_tk_image(denoised_img, size=(200, 200))
            if tk_img:
                img_label = tk.Label(card, image=tk_img, bg="white", relief="solid", borderwidth=1, bd=1)
                img_label.pack(pady=6, padx=6)
                self.image_refs.append(tk_img)
        else:
            tk.Label(card, text="Failed", bg="white", fg="red", width=25, height=10).pack(pady=6)
        
        if highlighted_img is not None:
            ctk.CTkLabel(
                card, 
                text="✨ Highlighted Changes", 
                font=("Arial", 10, "bold"), 
                text_color="#64748b"
            ).pack(pady=(8, 2))
            tk_high = self._to_tk_image(highlighted_img, size=(200, 200))
            if tk_high:
                high_label = tk.Label(card, image=tk_high, bg="white", relief="solid", borderwidth=1, bd=1)
                high_label.pack(pady=6, padx=6)
                self.image_refs.append(tk_high)
        
        if ssim_score != "-":
            color = "#10b981" if ssim_score >= 0.8 else "#f59e0b" if ssim_score >= 0.7 else "#ef4444"
            
            score_frame = ctk.CTkFrame(card, fg_color=("#f3f4f6", "#1f2937"), corner_radius=8)
            score_frame.pack(fill="x", padx=6, pady=6)
            
            ctk.CTkLabel(
                score_frame, 
                text=f"SSIM: {ssim_score:.4f}", 
                font=("Arial", 11, "bold"), 
                text_color=color
            ).pack(pady=4)
            
            ctk.CTkLabel(
                score_frame, 
                text=f"⏱️ {elapsed:.2f}s", 
                font=("Arial", 9), 
                text_color="#64748b"
            ).pack(pady=(0, 4))
        
        self.display_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _to_tk_image(self, img, size=(200, 200)):
        try:
            if img is None:
                return None
            arr = np.asarray(img)
            if arr.ndim == 3 and arr.shape[2] == 3:
                display = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            elif arr.ndim == 2:
                display = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            else:
                display = arr
            
            pil_img = Image.fromarray(display)
            pil_img.thumbnail(size, Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(pil_img)
        except Exception as e:
            print("Image conversion error:", e)
            return None
    
    def _set_busy(self, busy):
        state = "disabled" if busy else "normal"
        try:
            self.load_btn.configure(state=state)
            self.process_btn.configure(state=state)
        except Exception:
            pass

def main():
    root = tk.Tk()
    app = ImageDenoiserUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
