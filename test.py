import cv2

print("Testing camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not found!")
    print("Try these:")
    for i in range(3):
        test_cap = cv2.VideoCapture(i)
        if test_cap.isOpened():
            print(f"✅ Camera found at index {i}")
            test_cap.release()
else:
    print("✅ Camera working!")
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Camera Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
cap.release()
cv2.destroyAllWindows()