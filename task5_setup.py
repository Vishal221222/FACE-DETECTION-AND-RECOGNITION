
# Task 5 Setup Script
# Install required dependencies for Face Detection and Recognition

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def check_opencv():
    """Check if OpenCV is working"""
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
        return True
    except ImportError:
        print("❌ OpenCV not available")
        return False

def main():
    print("🚀 Task 5: Face Detection Setup")
    print("=" * 50)

    # Required packages
    packages = [
        "opencv-python",
        "numpy", 
        "pillow",
        "face-recognition"  # Optional but recommended
    ]

    print("📦 Installing required packages...")

    for package in packages:
        print(f"\nInstalling {package}...")
        install_package(package)

    print("\n🔍 Testing installation...")

    # Test OpenCV
    if check_opencv():
        try:
            import cv2
            # Test cascade classifiers
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            print("✅ Haar cascades available")
        except Exception as e:
            print(f"❌ Haar cascades error: {e}")

    # Test face_recognition
    try:
        import face_recognition
        print("✅ Face recognition library available")
    except ImportError:
        print("⚠️ Face recognition library not available (optional)")
        print("   Install with: pip install face_recognition")

    # Create face database directory
    if not os.path.exists("face_database"):
        os.makedirs("face_database")
        print("✅ Created face_database directory")

    print("\n🎯 Setup complete!")
    print("\nTo run the applications:")
    print("  python task5_face_detection.py")
    print("  python task5_face_detection_gui.py")

if __name__ == "__main__":
    main()
