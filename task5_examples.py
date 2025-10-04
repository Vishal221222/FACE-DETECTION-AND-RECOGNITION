
# Task 5: Face Detection Examples
# Usage examples for the face detection system

from task5_face_detection import FaceDetectionSystem
import cv2

def example_1_basic_detection():
    """Example 1: Basic face detection in an image"""
    print("Example 1: Basic Face Detection")
    print("=" * 40)

    # Create detector
    detector = FaceDetectionSystem()

    # Process an image (replace with your image path)
    image_path = "sample_image.jpg"

    # Uncomment to test with actual image:
    # result_image, results = detector.process_image(image_path)
    # 
    # if result_image is not None:
    #     print(f"Faces detected: {results['faces_detected']}")
    #     cv2.imshow('Face Detection', result_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # else:
    #     print(f"Error: {results}")

    print("Replace 'sample_image.jpg' with your image path")

def example_2_webcam_detection():
    """Example 2: Real-time webcam face detection"""
    print("Example 2: Webcam Detection")
    print("=" * 40)

    detector = FaceDetectionSystem()

    print("Starting webcam detection...")
    print("Press 'q' to quit, 's' to save frame")

    # Uncomment to start webcam:
    # detector.start_webcam_detection()

def example_3_face_recognition():
    """Example 3: Face recognition with database"""
    print("Example 3: Face Recognition")
    print("=" * 40)

    detector = FaceDetectionSystem()

    # Add a known face to database
    # Uncomment and replace paths:
    # success, message = detector.add_known_face("person1.jpg", "John Doe")
    # print(f"Add face result: {message}")

    # Process image with recognition
    # result_image, results = detector.process_image("group_photo.jpg", recognize=True)
    # print(f"Recognition results: {results.get('recognition', [])}")

def example_4_batch_processing():
    """Example 4: Batch process multiple images"""
    print("Example 4: Batch Processing")
    print("=" * 40)

    detector = FaceDetectionSystem()

    # List of images to process
    image_paths = [
        "image1.jpg",
        "image2.jpg", 
        "image3.jpg"
    ]

    results_summary = []

    for image_path in image_paths:
        print(f"Processing: {image_path}")

        # Uncomment to process:
        # result_image, results = detector.process_image(image_path)
        # if result_image is not None:
        #     results_summary.append({
        #         'image': image_path,
        #         'faces': results['faces_detected']
        #     })
        #     
        #     # Save result
        #     output_path = f"detected_{image_path}"
        #     cv2.imwrite(output_path, result_image)

    # Print summary
    # for result in results_summary:
    #     print(f"{result['image']}: {result['faces']} faces")

def example_5_gui_application():
    """Example 5: Launch GUI application"""
    print("Example 5: GUI Application")
    print("=" * 40)

    print("To launch the GUI application:")
    print("python task5_face_detection_gui.py")

    # Uncomment to launch GUI:
    # from task5_face_detection_gui import FaceDetectionGUI
    # app = FaceDetectionGUI()
    # app.run()

if __name__ == "__main__":
    print("🎯 Task 5: Face Detection Examples")
    print("=" * 50)
    print("Available examples:")
    print("1. example_1_basic_detection()")
    print("2. example_2_webcam_detection()")
    print("3. example_3_face_recognition()")
    print("4. example_4_batch_processing()")
    print("5. example_5_gui_application()")
    print()
    print("💡 Tips:")
    print("- Make sure to run task5_setup.py first")
    print("- Replace sample paths with your actual image paths")
    print("- Uncomment the code sections you want to test")
