# app.py
import os
import re
import uuid
import datetime
import hashlib
from io import BytesIO

import streamlit as st
from PIL import Image

try:
    import vertexai
    from vertexai.preview.vision_models import ImageGenerationModel
    from vertexai.generative_models import GenerativeModel, Part
    from google.oauth2 import service_account
    VERTEX_AVAILABLE = True
except Exception:
    VERTEX_AVAILABLE = False

# ---------------- Page config ----------------
st.set_page_config(page_title="AI Image Generator + Editor", layout="wide")
st.title("AI Image Generator + Editor")

# ---------------- Session initialization ----------------
def safe_init_session():
    try:
        _ = st.session_state
    except RuntimeError:
        return False
    st.session_state.setdefault("generated_images", [])   # list of {"filename","content","key","enhanced_prompt"}
    st.session_state.setdefault("edited_images", [])      # list of {"original","edited","prompt","filename","ts"}
    st.session_state.setdefault("edit_image_bytes", None) # bytes of the image currently loaded in the left editor
    st.session_state.setdefault("edit_image_name", "")
    st.session_state.setdefault("edit_iterations", 0)
    st.session_state.setdefault("max_edit_iterations", 20) # configurable cap
    return True

safe_init_session()

# ----------------- Embedded logo config -----------------
# Change this path to the logo file you want embedded into the app.
# Preferably PNG with transparency.
LOGO_PATH = "Dr._Reddy's_Laboratories_logo.svg.png"  # update if needed

def load_embedded_logo(path=LOGO_PATH):
    try:
        with open(path, "rb") as f:
            data = f.read()
            Image.open(BytesIO(data)).convert("RGBA")  # validate
            return data
    except Exception:
        return None

EMBEDDED_LOGO_BYTES = load_embedded_logo()

# ---------------- Prompt templates and style map ----------------
PROMPT_TEMPLATES = {
    "None": """
Dont make any changes in the user's prompt.Follow it as it is
User’s raw prompt:
"{USER_PROMPT}"

Refined general image prompt:
""",
    "General": """
You are an expert AI prompt engineer specialized in creating vivid and descriptive image prompts.

Your job:
- Expand the user’s input into a detailed, clear prompt for an image generation model.
- Add missing details such as:
  • Background and setting
  • Lighting and mood
  • Style and realism level
  • Perspective and composition

Rules:
- Stay true to the user’s intent.
- Keep language concise, descriptive, and expressive.
- Output only the final refined image prompt.

User’s raw prompt:
"{USER_PROMPT}"

Refined general image prompt:
""",
    # ... (other templates unchanged) ...
    "Marketing": """
You are a senior AI prompt engineer creating polished prompts for marketing and advertising visuals.

Task:
- Take the user’s raw input and turn it into a polished, professional, campaign-ready image prompt.
- Expand the idea with rich marketing-oriented details that make it visually persuasive.

When refining, include:
- Background & setting (modern, lifestyle, commercial, aspirational)
- Lighting & atmosphere (studio lights, golden hour, cinematic)
- Style (photorealistic, cinematic, product photography, lifestyle branding)
- Perspective & composition (wide shot, close-up, dramatic angles)
- Mood, tone & branding suitability (premium, sleek, aspirational)

Special Brand Rule:
- If the user asks for an image related to a specific brand, seamlessly add the brand’s tagline into the final image prompt.
- For **Dr. Reddy’s**, the correct tagline is: “Good Health Can’t Wait.”

Rules:
- Stay faithful to the user’s idea but elevate it for use in ads, social media, or presentations.
- Output **only** the final refined image prompt (no explanations, no extra text).

User raw input:
{USER_PROMPT}


Refined marketing image prompt:
""",
    # include other templates as needed...
}

STYLE_DESCRIPTIONS = {
    "None": "No special styling — keep the image natural, faithful to the user’s idea.",
    "Smart": "A clean, balanced, and polished look. Professional yet neutral, visually appealing without strong artistic bias.",
    "Cinematic": "Film-style composition with professional lighting. Wide dynamic range, dramatic highlights, storytelling feel.",
    # ... (rest omitted for brevity; copy as needed) ...
}

# ---------------- Helpers ----------------
def sanitize_prompt(text: str) -> str:
    if not text:
        return text
    lines = []
    for line in text.splitlines():
        ln = line.strip()
        if not ln:
            continue
        if re.match(r'^(Option|Key|Apply|Specificity|Keywords)\b', ln, re.I):
            continue
        if re.match(r'^\d+[\.\)]\s*', ln):
            continue
        if len(ln) < 80 and ln.endswith(':'):
            continue
        if ln.startswith('-') or ln.startswith('*'):
            continue
        lines.append(ln)
    cleaned = ' '.join(lines)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned or text

def safe_get_enhanced_text(resp):
    if resp is None:
        return ""
    if hasattr(resp, "text") and resp.text:
        return resp.text
    if hasattr(resp, "candidates") and resp.candidates:
        try:
            return resp.candidates[0].content.parts[0].text
        except Exception:
            pass
    return str(resp)

def get_image_bytes_from_genobj(gen_obj):
    if isinstance(gen_obj, (bytes, bytearray)):
        return bytes(gen_obj)
    for attr in ("image_bytes", "_image_bytes"):
        if hasattr(gen_obj, attr):
            return getattr(gen_obj, attr)
    if hasattr(gen_obj, "image") and gen_obj.image:
        for attr in ("image_bytes", "_image_bytes"):
            if hasattr(gen_obj.image, attr):
                return getattr(gen_obj.image, attr)
    return None

def show_image_safe(image_source, caption="Image"):
    try:
        if isinstance(image_source, (bytes, bytearray)):
            st.image(image_source, caption=caption, use_container_width=True)
        elif isinstance(image_source, Image.Image):
            st.image(image_source, caption=caption, use_container_width=True)
        else:
            st.image(Image.open(BytesIO(image_source)), caption=caption, use_container_width=True)
    except TypeError:
        if isinstance(image_source, (bytes, bytearray)):
            st.image(image_source, caption=caption, use_column_width=True)
        else:
            st.image(Image.open(BytesIO(image_source)), caption=caption, use_column_width=True)
    except Exception as e:
        st.error(f"Failed to display image: {e}")

# ---------------- Vertex lazy loaders ----------------
MODEL_CACHE = {"imagen": None, "nano": None, "text": None}

def init_vertex(project_id, credentials_info, location="us-central1"):
    if not VERTEX_AVAILABLE:
        return False
    try:
        if getattr(vertexai, "_initialized", False):
            return True
    except Exception:
        pass
    try:
        creds = service_account.Credentials.from_service_account_info(dict(credentials_info))
        vertexai.init(project=project_id, location=location, credentials=creds)
        setattr(vertexai, "_initialized", True)
        return True
    except Exception as e:
        st.error(f"Vertex init failed: {e}")
        return False

def get_imagen_model():
    if MODEL_CACHE["imagen"]:
        return MODEL_CACHE["imagen"]
    if not VERTEX_AVAILABLE:
        return None
    try:
        MODEL_CACHE["imagen"] = ImageGenerationModel.from_pretrained("imagen-4.0-generate-001")
        return MODEL_CACHE["imagen"]
    except Exception as e:
        st.error(f"Failed to load Imagen model: {e}")
        return None

def get_nano_banana_model():
    if MODEL_CACHE["nano"]:
        return MODEL_CACHE["nano"]
    if not VERTEX_AVAILABLE:
        return None
    try:
        MODEL_CACHE["nano"] = GenerativeModel("gemini-2.5-flash-image")
        return MODEL_CACHE["nano"]
    except Exception as e:
        st.error(f"Failed to load Nano Banana model: {e}")
        return None

def get_text_model():
    if MODEL_CACHE["text"]:
        return MODEL_CACHE["text"]
    if not VERTEX_AVAILABLE:
        return None
    try:
        MODEL_CACHE["text"] = GenerativeModel("gemini-2.0-flash")
        return MODEL_CACHE["text"]
    except Exception as e:
        st.error(f"Failed to load text model: {e}")
        return None

# ---------------- Image size helpers ----------------
def fit_image_to_target(img: Image.Image, target_size):
    """
    Resize img to cover the target_size and then center-crop to target_size.
    Keeps aspect ratio, avoids stretching.
    target_size = (w, h)
    """
    target_w, target_h = target_size
    if img.size == (target_w, target_h):
        return img

    img_w, img_h = img.size
    scale = max(target_w / img_w, target_h / img_h)
    new_w = int(img_w * scale + 0.5)
    new_h = int(img_h * scale + 0.5)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h
    img_cropped = img_resized.crop((left, top, right, bottom))

    return img_cropped

# ---------------- Core flows ----------------
def generate_images_from_prompt(prompt, dept="None", style_desc="", n_images=1):
    enhanced_prompt = prompt
    if not VERTEX_AVAILABLE:
        st.warning("VertexAI SDK not available — generation disabled in this environment.")
        return [], enhanced_prompt

    creds = st.secrets.get("gcp_service_account")
    if not creds or not creds.get("project_id"):
        st.warning("Missing GCP credentials in Streamlit secrets: 'gcp_service_account'. Generation disabled.")
        return [], enhanced_prompt

    if not init_vertex(creds["project_id"], creds):
        st.warning("Failed to initialize VertexAI.")
        return [], enhanced_prompt

    if dept and dept != "None":
        text_model = get_text_model()
        if text_model:
            try:
                template = PROMPT_TEMPLATES.get(dept, PROMPT_TEMPLATES["General"])
                refinement_input = template.replace("{USER_PROMPT}", prompt)
                if style_desc:
                    refinement_input += f"\n\nApply style: {style_desc}"
                text_resp = text_model.generate_content(refinement_input)
                maybe = safe_get_enhanced_text(text_resp).strip()
                cleaned = sanitize_prompt(maybe)
                if cleaned:
                    enhanced_prompt = cleaned
            except Exception as e:
                st.warning(f"Prompt refinement failed, using raw prompt. ({e})")
                enhanced_prompt = prompt

    imagen = get_imagen_model()
    if imagen is None:
        st.warning("Imagen model unavailable.")
        return [], enhanced_prompt

    try:
        resp = imagen.generate_images(prompt=enhanced_prompt, number_of_images=n_images)
    except Exception as e:
        st.error(f"Imagen generate_images failed: {e}")
        return [], enhanced_prompt

    out = []
    for i in range(min(n_images, len(resp.images))):
        gen_obj = resp.images[i]
        b = get_image_bytes_from_genobj(gen_obj)
        if b:
            out.append(b)
    return out, enhanced_prompt

def run_edit_flow(edit_prompt, base_bytes, enforce_canvas=True):
    """
    Use Nano Banana (Gemini image gen) to apply edits to base_bytes.
    - enforce_canvas: instruct model not to change canvas and post-process returned image to match original base image dimensions.
    Returns edited bytes or None.
    """
    if not VERTEX_AVAILABLE:
        st.warning("VertexAI SDK not available — editing disabled.")
        return None

    creds = st.secrets.get("gcp_service_account")
    if not creds or not creds.get("project_id"):
        st.warning("Missing GCP credentials in Streamlit secrets: 'gcp_service_account'. Editing disabled.")
        return None

    if not init_vertex(creds["project_id"], creds):
        st.warning("Failed to initialize VertexAI.")
        return None

    nano = get_nano_banana_model()
    if nano is None:
        st.warning("Nano Banana editor model unavailable.")
        return None

    try:
        base_img = Image.open(BytesIO(base_bytes)).convert("RGBA")
    except Exception as e:
        st.error(f"Failed to open base image: {e}")
        return None
    base_w, base_h = base_img.size

    strong_instruction = f"""
You are a professional AI image editor.
Instructions:
- Take the provided image and apply these edits: {edit_prompt}
- IMPORTANT: Do NOT change the canvas size, aspect ratio, or overall framing. Keep image dimensions exactly {base_w}x{base_h} pixels.
- Do NOT re-generate the scene or extend the canvas. Only edit pixels within the provided image frame.
- Return the final edited image inline (PNG). Do not include any extra text or captions.
"""
    input_image = Part.from_data(mime_type="image/png", data=base_bytes)

    try:
        response = nano.generate_content([strong_instruction, input_image])
    except Exception as e:
        st.error(f"Nano Banana call failed: {e}")
        return None

    edited_bytes = None
    for candidate in getattr(response, "candidates", []):
        for part in getattr(candidate.content, "parts", []):
            if hasattr(part, "inline_data") and getattr(part.inline_data, "data", None):
                edited_bytes = part.inline_data.data
                break
        if edited_bytes:
            break

    if not edited_bytes:
        if hasattr(response, "text") and response.text:
            st.warning("Editor returned text instead of an image. Check response.")
        else:
            st.warning("Editor returned no inline image.")
        return None

    # enforce canvas size if desired
    if enforce_canvas:
        try:
            edited_img = Image.open(BytesIO(edited_bytes)).convert("RGBA")
            if edited_img.size != (base_w, base_h):
                fixed = fit_image_to_target(edited_img, (base_w, base_h))
                buf = BytesIO()
                fixed.save(buf, format="PNG")
                return buf.getvalue()
            else:
                return edited_bytes
        except Exception as e:
            st.warning(f"Failed to post-process edited image to base canvas: {e}. Returning model output.")
            return edited_bytes

    return edited_bytes

def run_logo_mode(prompt_text, logo_bytes, base_bytes=None, scale=0.08, opacity=1.0,
                  placement_hint="bottom-right", use_gemini_for_blend=False):
    """
    Deterministically composite `logo_bytes` onto `base_bytes` using Pillow.
    - scale: fraction of image width for logo (e.g. 0.08 for 8%)
    - placement_hint: "top-left", "top-right", "bottom-left", "bottom-right", "center", or "x,y" (pixels)
    - use_gemini_for_blend: optional — send the composite to Gemini to subtly refine lighting while enforcing canvas size.
    Returns PNG bytes or None.
    """
    # If no base image provided, fall back to generation path (handled elsewhere)
    if base_bytes is None:
        st.info("No base image provided — use generation flow instead.")
        return None

    try:
        base_img = Image.open(BytesIO(base_bytes)).convert("RGBA")
    except Exception as e:
        st.error(f"Failed to open base image: {e}")
        return None
    base_w, base_h = base_img.size

    try:
        logo_img = Image.open(BytesIO(logo_bytes)).convert("RGBA")
    except Exception as e:
        st.error(f"Failed to open embedded logo: {e}")
        return None

    # Resize logo to fraction of base width
    target_w = max(1, int(base_w * float(scale)))
    wpercent = target_w / float(logo_img.width)
    target_h = int(float(logo_img.height) * wpercent)
    logo_resized = logo_img.resize((target_w, target_h), Image.LANCZOS)

    # Apply opacity
    if opacity < 0.999:
        alpha = logo_resized.split()[3]
        alpha = alpha.point(lambda p: int(p * opacity))
        logo_resized.putalpha(alpha)

    # placement logic
    margin = int(base_w * 0.02)
    ph = (placement_hint or "").strip().lower()
    if ph in ("top-left", "topleft", "top left"):
        pos = (margin, margin)
    elif ph in ("top-right", "topright", "top right"):
        pos = (base_w - logo_resized.width - margin, margin)
    elif ph in ("bottom-left", "bottomleft", "bottom left"):
        pos = (margin, base_h - logo_resized.height - margin)
    elif ph in ("bottom-right", "bottomright", "bottom right"):
        pos = (base_w - logo_resized.width - margin, base_h - logo_resized.height - margin)
    elif ph in ("center", "centre", "middle"):
        pos = ((base_w - logo_resized.width)//2, (base_h - logo_resized.height)//2)
    else:
        try:
            if "," in ph:
                x, y = [int(v.strip()) for v in ph.split(",")[:2]]
                pos = (x, y)
            else:
                pos = (base_w - logo_resized.width - margin, base_h - logo_resized.height - margin)
        except Exception:
            pos = (base_w - logo_resized.width - margin, base_h - logo_resized.height - margin)

    # Composite locally
    composite = base_img.copy()
    composite.paste(logo_resized, pos, logo_resized)

    buf = BytesIO()
    composite.save(buf, format="PNG")
    comp_bytes = buf.getvalue()

    # Optionally send to Gemini for subtle blending while enforcing canvas size after return
    if use_gemini_for_blend and VERTEX_AVAILABLE:
        nano = get_nano_banana_model()
        if nano is None:
            st.warning("Nano Banana unavailable, returning deterministic composite.")
            return comp_bytes

        instruction = f"""
You are a professional image editor.
Task:
- Improve lighting and blending on the provided composited image.
- DO NOT change canvas size, aspect ratio, or composition. Keep dimensions exactly {base_w}x{base_h}.
- Do NOT re-generate or crop; only apply subtle color / lighting / blending adjustments.
- Return the final PNG inline (no extra text).
"""
        comp_part = Part.from_data(mime_type="image/png", data=comp_bytes)
        try:
            response = nano.generate_content([instruction, comp_part])
        except Exception as e:
            st.warning(f"Nano Banana blending call failed ({e}); returning deterministic composite.")
            return comp_bytes

        edited_bytes = None
        for candidate in getattr(response, "candidates", []):
            for part in getattr(candidate.content, "parts", []):
                if hasattr(part, "inline_data") and getattr(part.inline_data, "data", None):
                    edited_bytes = part.inline_data.data
                    break
            if edited_bytes:
                break

        if not edited_bytes:
            st.warning("Nano Banana blending returned no inline image; returning deterministic composite.")
            return comp_bytes

        # Enforce canvas size
        try:
            edited_img = Image.open(BytesIO(edited_bytes)).convert("RGBA")
            if edited_img.size != (base_w, base_h):
                fixed = fit_image_to_target(edited_img, (base_w, base_h))
                buf2 = BytesIO()
                fixed.save(buf2, format="PNG")
                return buf2.getvalue()
            else:
                return edited_bytes
        except Exception as e:
            st.warning(f"Failed to enforce canvas after Gemini blend: {e}. Returning composite.")
            return comp_bytes

    return comp_bytes

# ---------------- UI ----------------
left_col, right_col = st.columns([3,1])

with left_col:
    # Controls
    dept = st.selectbox("🏢 Department ", list(PROMPT_TEMPLATES.keys()), index=0)
    style = st.selectbox("🎨 Style ", list(STYLE_DESCRIPTIONS.keys()), index=0)
    style_desc = "" if style == "None" else STYLE_DESCRIPTIONS.get(style, "")

    # Editor upload
    uploaded_file = st.file_uploader("Upload an image to edit ", type=["png","jpg","jpeg","webp"])
    if uploaded_file:
        raw = uploaded_file.read()
        pil = Image.open(BytesIO(raw)).convert("RGB")
        buf = BytesIO()
        pil.save(buf, format="PNG")
        buf_bytes = buf.getvalue()
        st.session_state["edit_image_bytes"] = buf_bytes
        st.session_state["edit_image_name"] = getattr(uploaded_file, "name", f"uploaded_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        st.session_state["edit_iterations"] = 0
        show_image_safe(buf_bytes, caption=f"Uploaded: {st.session_state['edit_image_name']}")
    else:
        if st.session_state.get("edit_image_bytes"):
            show_image_safe(st.session_state["edit_image_bytes"], caption=f"Editor loaded: {st.session_state.get('edit_image_name','Selected Image')}")

    prompt = st.text_area("Enter prompt", key="main_prompt", height=140, placeholder="")
    logo_mode = st.checkbox("Add logo", value=False)
    if logo_mode and not EMBEDDED_LOGO_BYTES:
        st.error("Embedded logo not found at LOGO_PATH. Please update LOGO_PATH or put the logo file there.")
        logo_mode = False

    # defaults
    placement_hint = ""
    use_gemini = False
    scale_frac = 0.08
    opacity = 1.0

    if logo_mode:
        st.markdown("**Logo settings (embedded logo)**")
        placement_hint = st.text_input("Placement hint (e.g. top-left, bottom-right, 'on car door', or 'x,y')", value="bottom-right")
        scale_percent = st.slider("Logo scale (% of image width)", min_value=1, max_value=40, value=8)
        opacity = st.slider("Logo opacity", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
        use_gemini = st.checkbox("Use Nano Banana for subtle blending (optional)", value=False)
        scale_frac = scale_percent / 100.0

    num_images = 1

    if st.button("Run"):
        prompt_text = (prompt or "").strip()
        if not prompt_text:
            st.warning("Please enter a prompt before running.")
        else:
            base_image = st.session_state.get("edit_image_bytes")

            # Logo mode (with deterministic pre-composite)
            if logo_mode and EMBEDDED_LOGO_BYTES:
                if not base_image:
                    st.error("To add an embedded logo onto an uploaded image you must upload a base image first.")
                else:
                    with st.spinner("Adding logo (deterministic composite)..."):
                        out_bytes = run_logo_mode(
                            prompt_text,
                            logo_bytes=EMBEDDED_LOGO_BYTES,
                            base_bytes=base_image,
                            scale=scale_frac,
                            opacity=opacity,
                            placement_hint=placement_hint,
                            use_gemini_for_blend=use_gemini
                        )
                        if out_bytes:
                            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            out_fn = f"outputs/logo/logo_{ts}_{uuid.uuid4().hex[:6]}.png"
                            os.makedirs(os.path.dirname(out_fn), exist_ok=True)
                            with open(out_fn, "wb") as f:
                                f.write(out_bytes)

                            st.success("Logo-mode image created.")
                            show_image_safe(out_bytes, caption=f"Logo Mode Result ({ts})")

                            current_name = os.path.basename(out_fn)
                            safe_key = hashlib.sha1(current_name.encode()).hexdigest()[:12]

                            col_dl, col_edit, col_clear = st.columns([1,1,1])
                            with col_dl:
                                st.download_button(
                                    "⬇️ Download (logo result)",
                                    data=out_bytes,
                                    file_name=current_name,
                                    mime="image/png",
                                    key=f"dl_logo_{safe_key}"
                                )
                            with col_edit:
                                if st.button("✏️ Edit this image ", key=f"edit_logo_{safe_key}"):
                                    st.session_state["edit_image_bytes"] = out_bytes
                                    st.session_state["edit_image_name"] = current_name
                                    st.session_state["edit_iterations"] = 0
                                    st.experimental_rerun()

                            st.session_state["edit_image_bytes"] = out_bytes
                            st.session_state["edit_image_name"] = current_name

                            st.session_state["edit_iterations"] = st.session_state.get("edit_iterations", 0) + 1
                            st.session_state.edited_images.append({
                                "original": base_image,
                                "edited": out_bytes,
                                "prompt": prompt_text,
                                "filename": out_fn,
                                "ts": ts
                            })

                            if st.session_state["edit_iterations"] >= st.session_state.get("max_edit_iterations", 20):
                                st.warning(f"Reached {st.session_state['edit_iterations']} edits. Please finalize or reset to avoid runaway costs.")
                        else:
                            st.error("Logo mode failed or returned no image.")
            else:
                # Standard edit (if base image present) — enforced canvas
                if base_image:
                    with st.spinner("Editing image..."):
                        edited = run_edit_flow(prompt_text, base_image, enforce_canvas=True)
                        if edited:
                            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            out_fn = f"outputs/edited/edited_{ts}_{uuid.uuid4().hex[:6]}.png"
                            os.makedirs(os.path.dirname(out_fn), exist_ok=True)
                            with open(out_fn, "wb") as f:
                                f.write(edited)

                            st.success("Edited image created.")
                            show_image_safe(edited, caption=f"Edited ({ts})")

                            current_name = os.path.basename(out_fn)
                            safe_key = hashlib.sha1(current_name.encode()).hexdigest()[:12]

                            col_dl, col_edit, col_clear = st.columns([1,1,1])
                            with col_dl:
                                st.download_button(
                                    "⬇️ Download Edited (current)",
                                    data=edited,
                                    file_name=current_name,
                                    mime="image/png",
                                    key=f"dl_edit_{safe_key}"
                                )
                            with col_edit:
                                if st.button("✏️ Edit this image ", key=f"edit_current_{safe_key}"):
                                    st.session_state["edit_image_bytes"] = edited
                                    st.session_state["edit_image_name"] = current_name
                                    st.session_state["edit_iterations"] = 0
                                    st.experimental_rerun()

                            st.session_state["edit_image_bytes"] = edited
                            st.session_state["edit_image_name"] = current_name

                            st.session_state["edit_iterations"] = st.session_state.get("edit_iterations", 0) + 1
                            st.session_state.edited_images.append({
                                "original": base_image,
                                "edited": edited,
                                "prompt": prompt_text,
                                "filename": out_fn,
                                "ts": ts
                            })

                            if st.session_state["edit_iterations"] >= st.session_state.get("max_edit_iterations", 20):
                                st.warning(f"Reached {st.session_state['edit_iterations']} edits. Please finalize or reset to avoid runaway costs.")
                        else:
                            st.error("Editing failed or returned no image.")
                else:
                    # GENERATION flow (Imagen)
                    with st.spinner("Generating images..."):
                        generated, enhanced = generate_images_from_prompt(prompt_text, dept=dept, style_desc=style_desc, n_images=num_images)
                        if generated:
                            st.success(f"Generated {len(generated)} image(s).")
                            for i, b in enumerate(generated):
                                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                fname = f"outputs/generated/gen_{ts}_{i}.png"
                                os.makedirs(os.path.dirname(fname), exist_ok=True)
                                with open(fname, "wb") as f:
                                    f.write(b)

                                short = os.path.basename(fname) + str(i)
                                key_hash = uuid.uuid5(uuid.NAMESPACE_DNS, short).hex[:8]
                                entry = {"filename": fname, "content": b, "key": key_hash, "enhanced_prompt": enhanced}
                                st.session_state.generated_images.append(entry)
                        else:
                            st.error("Generation failed or returned no images.")

    st.markdown("---")

    # -------------------------
    # Render Recently Generated
    # -------------------------
    if st.session_state.get("generated_images"):
        st.markdown("### Recently Generated")
        for entry in reversed(st.session_state.generated_images[-12:]):
            fname = entry.get("filename")
            b = entry.get("content")
            key_hash = entry.get("key") or uuid.uuid5(uuid.NAMESPACE_DNS, os.path.basename(fname)).hex[:8]
            enhanced_prompt = entry.get("enhanced_prompt", "")

            show_image_safe(b, caption=os.path.basename(fname))

            if enhanced_prompt:
                with st.expander("Enhanced prompt (refined)"):
                    st.code(enhanced_prompt)

            col_dl, col_edit = st.columns([1,1])
            with col_dl:
                st.download_button(
                    "⬇️ Download",
                    data=b,
                    file_name=os.path.basename(fname),
                    mime="image/png",
                    key=f"dl_gen_{key_hash}"
                )
            with col_edit:
                if st.button("✏️ Edit ", key=f"edit_gen_{key_hash}"):
                    st.session_state["edit_image_bytes"] = b
                    st.session_state["edit_image_name"] = os.path.basename(fname)
                    st.session_state["edit_iterations"] = 0
                    st.experimental_rerun()

    # -------------------------
    # Render Edited History
    # -------------------------
    if st.session_state.get("edited_images"):
        st.markdown("### Edited History (chain)")
        for idx, entry in enumerate(reversed(st.session_state.edited_images[-40:])):
            name = os.path.basename(entry.get("filename", f"edited_{idx}.png"))
            orig = entry.get("original")
            edited_bytes = entry.get("edited")
            prompt_prev = entry.get("prompt", "")
            ts = entry.get("ts", "")
            hash_k = hashlib.sha1((name + ts + str(idx)).encode()).hexdigest()[:12]

            with st.expander(f"{name} — {prompt_prev[:80]}"):
                col1, col2 = st.columns(2)
                with col1:
                    if orig:
                        show_image_safe(orig, caption="Before")
                with col2:
                    show_image_safe(edited_bytes, caption="After")

                col_dl, col_edit = st.columns([1,1])
                with col_dl:
                    st.download_button("⬇️ Download Edited", data=edited_bytes, file_name=name, mime="image/png", key=f"hist_dl_{hash_k}")
                with col_edit:
                    if st.button("✏️ Edit this version", key=f"hist_edit_{hash_k}"):
                        st.session_state["edit_image_bytes"] = edited_bytes
                        st.session_state["edit_image_name"] = name
                        st.session_state["edit_iterations"] = 0
                        st.experimental_rerun()

# ---------------- Right column: smaller history + controls ----------------
with right_col:
    max_it = 100
    st.session_state["max_edit_iterations"] = int(max_it)

    st.markdown("**Embedded logo status**")
    if EMBEDDED_LOGO_BYTES:
        show_image_safe(EMBEDDED_LOGO_BYTES, caption="Embedded logo (used when Logo Mode enabled)")
    else:
        st.warning(f"No embedded logo found at '{LOGO_PATH}'. Update LOGO_PATH at top of file.")
