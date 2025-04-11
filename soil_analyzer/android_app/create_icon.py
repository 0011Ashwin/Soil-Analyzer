from PIL import Image, ImageDraw, ImageFont
import os

def create_soil_icon(output_path, size=(512, 512), bg_color=(0, 163, 108), fg_color=(255, 255, 255)):
    """Create a simple icon for the Soil Health Analyzer app"""
    # Create a base image with the background color
    img = Image.new('RGB', size, bg_color)
    draw = ImageDraw.Draw(img)
    
    # Draw soil layers
    width, height = size
    
    # Dark soil layer
    draw.rectangle([(width * 0.1, height * 0.6), (width * 0.9, height * 0.85)], 
                   fill=(101, 67, 33))  # Brown
    
    # Mid soil layer
    draw.rectangle([(width * 0.1, height * 0.5), (width * 0.9, height * 0.6)], 
                   fill=(139, 69, 19))  # Saddle brown
    
    # Top soil layer
    draw.rectangle([(width * 0.1, height * 0.4), (width * 0.9, height * 0.5)], 
                   fill=(160, 82, 45))  # Sienna
    
    # Draw a plant
    # Stem
    draw.rectangle([(width * 0.5 - width * 0.02, height * 0.2), (width * 0.5 + width * 0.02, height * 0.4)], 
                   fill=(0, 128, 0))  # Green
    
    # Leaves (simple triangles)
    # Left leaf
    points = [
        (width * 0.5, height * 0.25),  # Stem connection
        (width * 0.3, height * 0.2),   # Tip of leaf
        (width * 0.4, height * 0.3)    # Base of leaf
    ]
    draw.polygon(points, fill=(0, 128, 0))  # Green
    
    # Right leaf
    points = [
        (width * 0.5, height * 0.25),  # Stem connection
        (width * 0.7, height * 0.2),   # Tip of leaf
        (width * 0.6, height * 0.3)    # Base of leaf
    ]
    draw.polygon(points, fill=(0, 128, 0))  # Green
    
    # Add another set of leaves
    # Left leaf
    points = [
        (width * 0.5, height * 0.35),  # Stem connection
        (width * 0.3, height * 0.3),   # Tip of leaf
        (width * 0.4, height * 0.4)    # Base of leaf
    ]
    draw.polygon(points, fill=(0, 128, 0))  # Green
    
    # Right leaf
    points = [
        (width * 0.5, height * 0.35),  # Stem connection
        (width * 0.7, height * 0.3),   # Tip of leaf
        (width * 0.6, height * 0.4)    # Base of leaf
    ]
    draw.polygon(points, fill=(0, 128, 0))  # Green
    
    # Draw a root system
    # Main root
    draw.rectangle([(width * 0.5 - width * 0.01, height * 0.4), (width * 0.5 + width * 0.01, height * 0.7)], 
                   fill=(139, 69, 19))  # Brown
    
    # Small roots
    for y_pos, angle in [(0.5, 30), (0.55, -30), (0.6, 25), (0.65, -25)]:
        start_x = width * 0.5
        start_y = height * y_pos
        end_x = start_x + width * 0.15 * (1 if angle > 0 else -1)
        end_y = start_y + height * 0.05
        
        # Draw a small root
        draw.line([(start_x, start_y), (end_x, end_y)], fill=(139, 69, 19), width=int(width * 0.01))
    
    # Add a circular border
    border_width = int(width * 0.03)
    draw.ellipse([(border_width, border_width), 
                 (width - border_width, height - border_width)], 
                 outline=fg_color, width=border_width)
    
    # Save the image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    
    print(f"Icon created at {output_path}")
    
    # Create different sizes for Android
    sizes = {
        'ldpi': (36, 36),
        'mdpi': (48, 48),
        'hdpi': (72, 72),
        'xhdpi': (96, 96),
        'xxhdpi': (144, 144),
        'xxxhdpi': (192, 192),
    }
    
    # Create a directory for each size
    base_dir = os.path.dirname(output_path)
    for density, size in sizes.items():
        density_dir = os.path.join(base_dir, f"drawable-{density}")
        os.makedirs(density_dir, exist_ok=True)
        
        # Resize and save the icon
        resized_img = img.resize(size, Image.LANCZOS)
        icon_path = os.path.join(density_dir, "icon.png")
        resized_img.save(icon_path)
        print(f"Created {density} icon at {icon_path}")

if __name__ == "__main__":
    # Create the app icon
    create_soil_icon("soil_analyzer/android_app/assets/icon.png") 