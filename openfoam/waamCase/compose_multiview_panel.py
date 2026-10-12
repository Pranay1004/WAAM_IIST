import os
from PIL import Image, ImageDraw, ImageFont

scratch_dir = "/Users/pandeyji/.gemini/antigravity-ide/brain/9931f6d8-ca6b-4074-a405-04685e7fa6fe/scratch/"

# Load 4 view images
im1 = Image.open(os.path.join(scratch_dir, "view1_isometric.png"))
im2 = Image.open(os.path.join(scratch_dir, "view2_top.png"))
im3 = Image.open(os.path.join(scratch_dir, "view3_side.png"))
im4 = Image.open(os.path.join(scratch_dir, "view4_front.png"))

W, H = im1.size # 1600 x 1000

# Crop tighter to focal areas
# view1_isometric: keep center and right (color bar)
crop1 = im1.crop((100, 150, 1550, 950)) # 1450 x 800

# view2_top: crop center bead
crop2 = im2.crop((300, 200, 1400, 850)) # 1100 x 650

# view3_side: crop horizontal strip
crop3 = im3.crop((100, 300, 1450, 750)) # 1350 x 450

# view4_front: crop transverse profile
crop4 = im4.crop((200, 320, 1450, 750)) # 1250 x 430

# Target composite size: 2x2 grid with header banners
target_w = 1200
target_h = 750

crop1 = crop1.resize((target_w, target_h), Image.Resampling.LANCZOS)
crop2 = crop2.resize((target_w, target_h), Image.Resampling.LANCZOS)
crop3 = crop3.resize((target_w, target_h), Image.Resampling.LANCZOS)
crop4 = crop4.resize((target_w, target_h), Image.Resampling.LANCZOS)

# Composite image: 2 columns, 2 rows + top title margin
canvas_w = target_w * 2 + 60
canvas_h = target_h * 2 + 100
composite = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
draw = ImageDraw.Draw(composite)

try:
    font_title = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 34)
    font_label = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 26)
    font_sub = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 20)
except Exception:
    font_title = font_label = font_sub = ImageFont.load_default()

# Paste images
# Row 0
composite.paste(crop1, (20, 70))
composite.paste(crop2, (target_w + 40, 70))
# Row 1
composite.paste(crop3, (20, target_h + 90))
composite.paste(crop4, (target_w + 40, target_h + 90))

# Main Title
draw.text((30, 20), "WAAM Multiphase CFD: Multi-View Deposited Bead Morphology (ER70S6 / OpenFOAM v2412)", fill=(10, 20, 40), font=font_title)

# Panel Labels
# Panel (a)
draw.rectangle([(30, 80), (480, 125)], fill=(255, 255, 255, 220), outline=(0, 0, 0), width=2)
draw.text((40, 85), "(a) 3D Isometric View (Perspective)", fill=(0, 0, 0), font=font_label)

# Panel (b)
draw.rectangle([(target_w + 50, 80), (target_w + 420, 125)], fill=(255, 255, 255, 220), outline=(0, 0, 0), width=2)
draw.text((target_w + 60, 85), "(b) Top-Down Plan View (XY)", fill=(0, 0, 0), font=font_label)

# Panel (c)
draw.rectangle([(30, target_h + 100), (510, target_h + 145)], fill=(255, 255, 255, 220), outline=(0, 0, 0), width=2)
draw.text((40, target_h + 105), "(c) Longitudinal Elevation (XZ @ y=0)", fill=(0, 0, 0), font=font_label)

# Panel (d)
draw.rectangle([(target_w + 50, target_h + 100), (target_w + 480, target_h + 145)], fill=(255, 255, 255, 220), outline=(0, 0, 0), width=2)
draw.text((target_w + 60, target_h + 105), "(d) Transverse Crown Profile (YZ)", fill=(0, 0, 0), font=font_label)

output_path = os.path.join(scratch_dir, "bead_multiview_panel.png")
composite.save(output_path, quality=95)
print("Composite saved successfully to:", output_path)
