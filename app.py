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

# ---------------- Embedded logo config (for post-generation overlay) ----------------
# Update this path to point at your embedded logo file (PNG recommended with transparency)
LOGO_PATH = "Dr._Reddy's_Laboratories_logo.svg.png"

def load_embedded_logo(path=LOGO_PATH):
    try:
        with open(path, "rb") as f:
            data = f.read()
            # validate image
            Image.open(BytesIO(data)).convert("RGBA")
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

    "Design": """
You are a senior AI prompt engineer supporting a creative design team.

Your job:
- Expand raw input into a visually inspiring, design-oriented image prompt.
- Add imaginative details about:
  • Artistic styles (minimalist, abstract, futuristic, flat, 3D render, watercolor, digital illustration)
  • Color schemes, palettes, textures, and patterns
  • Composition and balance (symmetry, negative space, creative framing)
  • Lighting and atmosphere (soft glow, vibrant contrast, surreal shading)
  • Perspective (isometric, top-down, wide shot, close-up)

Rules:
- Keep fidelity to the idea but make it highly creative and visually unique.
- Output only the final refined image prompt.

User’s raw prompt:
"{USER_PROMPT}"

Refined design image prompt:
""",
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
    "DPEX": """
You are a senior AI prompt engineer creating refined prompts for IT and technology-related visuals.

Your job:
- Transform the raw input into a detailed, professional, and technology-focused image prompt.
- Expand with contextual details about:
  • Technology environments (server rooms, data centers, cloud systems, coding workspaces)
  • Digital elements (network diagrams, futuristic UIs, holograms, cybersecurity visuals)
  • People in IT roles (developers, engineers, admins, tech support, collaboration)
  • Tone (innovative, technical, futuristic, professional)
  • Composition (screens, servers, code on monitors, abstract digital patterns)
  • Lighting and effects (LED glow, cyberpunk tones, neon highlights, modern tech ambiance)

Rules:
- Ensure images are suitable for IT presentations, product demos, training, technical documentation, and digital transformation campaigns.
- Stay true to the user’s intent but emphasize a technological and innovative look.
- Output only the final refined image prompt.

User’s raw prompt:
"{USER_PROMPT}"

Refined DPEX image prompt:
""",
    "HR": """
You are a senior AI prompt engineer creating refined prompts for human resources and workplace-related visuals.

Your job:
- Transform the raw input into a detailed, professional, and HR-focused image prompt.
- Expand with contextual details about:
  • Workplace settings (modern office, meeting rooms, open workspaces, onboarding sessions)
  • People interactions (interviews, teamwork, training, collaboration, diversity and inclusion)
  • Themes (employee engagement, professional growth, recruitment, performance evaluation)
  • Composition (groups in discussion, managers mentoring, collaborative workshops)
  • Lighting and tone (bright, welcoming, professional, inclusive)

Rules:
- Ensure images are suitable for HR presentations, recruitment campaigns, internal training, or employee engagement material.
- Stay true to the user’s intent but emphasize people, culture, and workplace positivity.
- Output only the final refined image prompt.

User’s raw prompt:
"{USER_PROMPT}"

Refined HR image prompt:
""",
    "Business": """
You are a senior AI prompt engineer creating refined prompts for business and corporate visuals.

Your job:
- Transform the raw input into a detailed, professional, and business-oriented image prompt.
- Expand with contextual details about:
  • Corporate settings (boardrooms, skyscrapers, modern offices, networking events)
  • Business activities (presentations, negotiations, brainstorming sessions, teamwork)
  • People (executives, entrepreneurs, consultants, diverse teams, global collaboration)
  • Tone (professional, ambitious, strategic, innovative)
  • Composition (formal meetings, handshake deals, conference tables, city skyline backgrounds)
  • Lighting and atmosphere (clean, modern, premium, professional)

Rules:
- Ensure images are suitable for corporate branding, investor decks, strategy sessions, or professional reports.
- Stay true to the user’s intent but emphasize professionalism, ambition, and success.
- Output only the final refined image prompt.

User’s raw prompt:
"{USER_PROMPT}"

Refined business image prompt:
"""
}


STYLE_DESCRIPTIONS = {
    "None": "No special styling — keep the image natural, faithful to the user’s idea.",
    "Smart": "A clean, balanced, and polished look. Professional yet neutral, visually appealing without strong artistic bias.",
    "Cinematic": "Film-style composition with professional lighting. Wide dynamic range, dramatic highlights, storytelling feel.",
    "Creative": "Playful, imaginative, and experimental. Bold artistic choices, unexpected elements, and expressive color use.",
    "Bokeh": "Photography style with shallow depth of field. Subject in sharp focus with soft, dreamy, blurred backgrounds.",
    "Macro": "Extreme close-up photography. High detail, textures visible, shallow focus highlighting minute features.",
    "Illustration": "Hand-drawn or digitally illustrated style. Clear outlines, stylized shading, expressive and artistic.",
    "3D Render": "Photorealistic or stylized CGI. Crisp geometry, depth, shadows, and reflective surfaces with realistic rendering.",
    "Fashion": "High-end editorial photography. Stylish, glamorous poses, bold makeup, controlled lighting, and modern aesthetic.",
    "Minimalist": "Simple and uncluttered. Few elements, large negative space, flat or muted color palette, clean composition.",
    "Moody": "Dark, atmospheric, and emotional. Strong shadows, high contrast, deep tones, cinematic ambiance.",
    "Portrait": "Focus on the subject. Natural skin tones, shallow depth of field, close-up or waist-up framing, studio or natural lighting.",
    "Stock Photo": "Professional, commercial-quality photo. Neutral subject matter, polished composition, business-friendly aesthetic.",
    "Vibrant": "Bold, saturated colors. High contrast, energetic mood, eye-catching and lively presentation.",
    "Pop Art": "Comic-book and pop-art inspired. Bold outlines, halftone patterns, flat vivid colors, high contrast, playful tone.",
    "Vector": "Flat vector graphics. Smooth shapes, sharp edges, solid fills, and clean scalable style like logos or icons.",

    "Watercolor": "Soft, fluid strokes with delicate blending and washed-out textures. Artistic and dreamy.",
    "Oil Painting": "Rich, textured brushstrokes. Classic fine art look with deep color blending.",
    "Charcoal": "Rough, sketchy textures with dark shading. Artistic, raw, dramatic effect.",
    "Line Art": "Minimal monochrome outlines with clean, bold strokes. No shading, focus on form.",

    "Anime": "Japanese animation style with vibrant colors, clean outlines, expressive features, and stylized proportions.",
    "Cartoon": "Playful, exaggerated features, simplified shapes, bold outlines, and bright colors.",
    "Pixel Art": "Retro digital art style. Small, pixel-based visuals resembling old-school video games.",

    "Fantasy Art": "Epic fantasy scenes. Magical elements, mythical creatures, enchanted landscapes.",
    "Surreal": "Dreamlike, bizarre imagery. Juxtaposes unexpected elements, bending reality.",
    "Concept Art": "Imaginative, detailed artwork for games or films. Often moody and cinematic.",

    "Cyberpunk": "Futuristic neon city vibes. High contrast, glowing lights, dark tones, sci-fi feel.",
    "Steampunk": "Retro-futuristic style with gears, brass, Victorian aesthetics, and industrial design.",
    "Neon Glow": "Bright neon outlines and glowing highlights. Futuristic, nightlife aesthetic.",
    "Low Poly": "Simplified 3D style using flat geometric shapes and polygons.",
    "Isometric": "3D look with isometric perspective. Often used for architecture, games, and diagrams.",

    "Vintage": "Old-school, retro tones. Faded colors, film grain, sepia, or retro print feel.",
    "Graffiti": "Urban street art style with bold colors, spray paint textures, and rebellious tone."
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

# ---------------- Logo overlay helper (post-generation) ----------------
def overlay_logo_on_image(image_bytes, logo_bytes, placement="bottom-right", scale=0.08, opacity=1.0):
    """Deterministically overlay embedded logo onto image bytes and return new PNG bytes."""
    try:
        base = Image.open(BytesIO(image_bytes)).convert("RGBA")
        logo = Image.open(BytesIO(logo_bytes)).convert("RGBA")
    except Exception as e:
        st.warning(f"Failed to open base/logo images for overlay: {e}")
        return image_bytes

    base_w, base_h = base.size
    # compute logo size (fraction of base width)
    new_logo_w = max(1, int(base_w * float(scale)))
    w_percent = new_logo_w / logo.width
    new_logo_h = int(logo.height * w_percent)
    logo = logo.resize((new_logo_w, new_logo_h), Image.LANCZOS)

    # apply opacity
    if opacity < 1.0:
        alpha = logo.split()[3].point(lambda p: int(p * opacity))
        logo.putalpha(alpha)

    margin = int(base_w * 0.02)
    ph = (placement or "").strip().lower()
    if "bottom" in ph:
        y = base_h - new_logo_h - margin
    elif "top" in ph:
        y = margin
    else:
        y = (base_h - new_logo_h) // 2

    if "right" in ph:
        x = base_w - new_logo_w - margin
    elif "left" in ph:
        x = margin
    else:
        x = (base_w - new_logo_w) // 2

    pos = (x, y)
    base.paste(logo, pos, logo)
    out = BytesIO()
    base.save(out, format="PNG")
    return out.getvalue()

# ---------------- Core flows ----------------
def generate_images_from_prompt(prompt, dept="None", style_desc="", n_images=1):
    """
    Returns (list_of_image_bytes, enhanced_prompt_str)
    """
    enhanced_prompt = prompt  # default

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

    # attempt text refinement when dept is selected
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

def run_edit_flow(edit_prompt, base_bytes):
    """
    Use Nano Banana (Gemini image gen) to apply edits to base_bytes.
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

    # Build parts: inline image + text instruction
    input_image = Part.from_data(mime_type="image/png", data=base_bytes)
    edit_instruction = f"""
You are a professional AI image editor.
Instructions:
- Take the provided image.
- Apply these edits: {edit_prompt}
- Return the final edited image inline (PNG).
- Do not include any extra text or captions.
"""
    try:
        response = nano.generate_content([edit_instruction, input_image])
    except Exception as e:
        st.error(f"Nano Banana call failed: {e}")
        return None

    for candidate in getattr(response, "candidates", []):
        for part in getattr(candidate.content, "parts", []):
            if hasattr(part, "inline_data") and getattr(part.inline_data, "data", None):
                return part.inline_data.data

    if hasattr(response, "text") and response.text:
        st.warning("Editor returned text instead of an image. Check response.")
    else:
        st.warning("Editor returned no inline image.")
    return None

# ---------------- UI ----------------
left_col, right_col = st.columns([3,1])

with left_col:
   
    # Controls
    dept = st.selectbox("🏢 Department ", list(PROMPT_TEMPLATES.keys()), index=0)
    style = st.selectbox("🎨 Style ", list(STYLE_DESCRIPTIONS.keys()), index=0)
    style_desc = "" if style == "None" else STYLE_DESCRIPTIONS.get(style, "")

    uploaded_file = st.file_uploader("Upload an image to edit ", type=["png","jpg","jpeg","webp"])
    if uploaded_file:
        raw = uploaded_file.read()
        pil = Image.open(BytesIO(raw)).convert("RGB")
        buf = BytesIO()
        pil.save(buf, format="PNG")
        buf_bytes = buf.getvalue()
        # immediately load uploaded image into editor
        st.session_state["edit_image_bytes"] = buf_bytes
        st.session_state["edit_image_name"] = getattr(uploaded_file, "name", f"uploaded_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        st.session_state["edit_iterations"] = 0
        show_image_safe(buf_bytes, caption=f"Uploaded: {st.session_state['edit_image_name']}")
    else:
        if st.session_state.get("edit_image_bytes"):
            show_image_safe(st.session_state["edit_image_bytes"], caption=f"Editor loaded: {st.session_state.get('edit_image_name','Selected Image')}")

    prompt = st.text_area("Enter prompt", key="main_prompt", height=140, placeholder="")
    
    # --- NEW: Post-generation logo mode controls ---
    logo_mode = st.checkbox("Add embedded logo after generation", value=False)
    if logo_mode and not EMBEDDED_LOGO_BYTES:
        st.error(f"Embedded logo not found at LOGO_PATH ('{LOGO_PATH}'). Update LOGO_PATH or add the file.")
        logo_mode = False

    placement_hint = "bottom-right"
    scale_frac = 0.08
    opacity = 1.0
    if logo_mode:
        st.markdown("Logo")
        placement_hint = st.text_input("Placement (top-left, top-right, bottom-left, bottom-right, center)", value="bottom-right")
        scale_percent = 20
        opacity = 1
        scale_frac = scale_percent / 100.0
                         
    num_images = 1

    # Run button: either Edit (if edit image loaded) or Generate
    if st.button("Run"):
        prompt_text = (prompt or "").strip()
        if not prompt_text:
            st.warning("Please enter a prompt before running.")
        else:
            base_image = st.session_state.get("edit_image_bytes")
            if base_image:
                # EDIT flow: edit the loaded image and make result the new loaded image
                with st.spinner("Editing image..."):
                    edited = run_edit_flow(prompt_text, base_image)
                    if edited:
                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        out_fn = f"outputs/edited/edited_{ts}_{uuid.uuid4().hex[:6]}.png"
                        os.makedirs(os.path.dirname(out_fn), exist_ok=True)
                        with open(out_fn, "wb") as f:
                            f.write(edited)

                        st.success("Edited image created.")
                        show_image_safe(edited, caption=f"Edited ({ts})")

                        # --------------------------
                        # NEW: 3-column controls for current edited image: Download | Edit (load into editor) | Clear
                        # --------------------------
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
                                # put the current edited bytes into the editor slot (so the next Run will edit this image)
                                st.session_state["edit_image_bytes"] = edited
                                st.session_state["edit_image_name"] = current_name
                                st.session_state["edit_iterations"] = 0
                                st.experimental_rerun()
                        
                        # --------------------------

                        # Replace the editor image with the freshly edited bytes so user can re-edit
                        st.session_state["edit_image_bytes"] = edited
                        st.session_state["edit_image_name"] = current_name

                        # increment iteration counter and append to edited history chain
                        st.session_state["edit_iterations"] = st.session_state.get("edit_iterations", 0) + 1
                        st.session_state.edited_images.append({
                            "original": base_image,
                            "edited": edited,
                            "prompt": prompt_text,
                            "filename": out_fn,
                            "ts": ts
                        })

                        # optional guard
                        if st.session_state["edit_iterations"] >= st.session_state.get("max_edit_iterations", 20):
                            st.warning(f"Reached {st.session_state['edit_iterations']} edits. Please finalize or reset to avoid runaway costs.")
                    else:
                        st.error("Editing failed or returned no image.")
            else:
                # GENERATION flow
                with st.spinner("Generating images..."):
                    generated, enhanced = generate_images_from_prompt(prompt_text, dept=dept, style_desc=style_desc, n_images=num_images)
                    if generated:
                        st.success(f"Generated {len(generated)} image(s).")
                        for i, b in enumerate(generated):
                            # Post-generation: overlay embedded logo if requested
                            out_bytes = b
                            if logo_mode and EMBEDDED_LOGO_BYTES:
                                try:
                                    out_bytes = overlay_logo_on_image(b, EMBEDDED_LOGO_BYTES, placement=placement_hint, scale=scale_frac, opacity=opacity)
                                except Exception as e:
                                    st.warning(f"Logo overlay failed, saving original generated image. ({e})")
                                    out_bytes = b

                            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            fname = f"outputs/generated/gen_{ts}_{i}.png"
                            os.makedirs(os.path.dirname(fname), exist_ok=True)
                            with open(fname, "wb") as f:
                                f.write(out_bytes)

                            short = os.path.basename(fname) + str(i)
                            key_hash = uuid.uuid5(uuid.NAMESPACE_DNS, short).hex[:8]
                            entry = {"filename": fname, "content": out_bytes, "key": key_hash, "enhanced_prompt": enhanced}
                            st.session_state.generated_images.append(entry)
                    else:
                        st.error("Generation failed or returned no images.")

    # Option to clear editor (go back to generate-mode)
    

    st.markdown("---")

    # -------------------------
    # Render Recently Generated (persistent, outside Run block)
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
                # stable download key per image
                st.download_button(
                    "⬇️ Download",
                    data=b,
                    file_name=os.path.basename(fname),
                    mime="image/png",
                    key=f"dl_gen_{key_hash}"
                )
            with col_edit:
                # stable edit button - loads the image into the editor so it becomes re-editable
                if st.button("✏️ Edit ", key=f"edit_gen_{key_hash}"):
                    st.session_state["edit_image_bytes"] = b
                    st.session_state["edit_image_name"] = os.path.basename(fname)
                    st.session_state["edit_iterations"] = 0
                    st.experimental_rerun()

    # -------------------------
    # Render Edited History (allow picking any previous edited version to continue)
    # -------------------------
    if st.session_state.get("edited_images"):
        st.markdown("### Edited History (chain)")
        for idx, entry in enumerate(reversed(st.session_state.edited_images[-40:])):
            name = os.path.basename(entry.get("filename", f"edited_{idx}.png"))
            orig = entry.get("original")
            edited_bytes = entry.get("edited")
            prompt_prev = entry.get("prompt", "")
            ts = entry.get("ts", "")
            # uniqueish key for widgets in this loop
            hash_k = hashlib.sha1((name + ts + str(idx)).encode()).hexdigest()[:12]

            with st.expander(f"{name} — {prompt_prev[:80]}"):
                col1, col2 = st.columns(2)
                with col1:
                    if orig:
                        show_image_safe(orig, caption="Before")
                with col2:
                    show_image_safe(edited_bytes, caption="After")

                # download and continue-edit buttons side-by-side
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

    
    if EMBEDDED_LOGO_BYTES:
        show_image_safe(EMBEDDED_LOGO_BYTES, caption="Embedded logo (post-generation)")
    else:
        st.warning(f"No embedded logo found at '{LOGO_PATH}'. Update LOGO_PATH at top of file.")
