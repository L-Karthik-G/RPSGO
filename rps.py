"""
Rock Paper Scissors - GPU Optimized Version
Uses a GPU (device='cuda:0') for ultra-fast YOLO inference,
and implements a game state machine to prevent frame freezing.
This version detects hands anywhere on the player's side of the screen.
"""
<<<<<<< HEAD

=======
>>>>>>> d06c91da98281b1556a468bfd83832f960081a67
import cv2
from ultralytics import YOLO
import sys
import time
from typing import Tuple, Optional, Dict


# ============== CONFIGURATION ==============
# Path to your YOLO model weights
MODEL_PATH = 'runs/detect/rps_optimized/weights/best.pt'
CONFIDENCE = 0.5 
INFERENCE_W, INFERENCE_H = 640, 480 
COUNTDOWN_TIME = 3.0 # Seconds for the countdown timer

# Colors assigned to each gesture for visual identification (BGR format)
GESTURE_COLORS: Dict[str, Tuple[int, int, int]] = {
    'rock': (10, 100, 200),     # Orange/Brown
    'paper': (255, 255, 0),    # Cyan/Light Blue
    'scissors': (255, 0, 255)  # Magenta/Purple
}

print("="*60)
print("STARTING GPU-OPTIMIZED GAME...")
print("="*60)
name_1 = input("Enter name for Player 1: ").strip() or "Player 1"
name_2 = input("Enter name for Player 2: ").strip() or "Player 2"

# ============== LOAD MODEL & DEVICE CHECK ==============
DEVICE = 'cpu'
try:
    import torch
    
    # 1. Determine the best available device
    if torch.cuda.is_available():
         DEVICE = 'cuda:0'
         print("🚀 Using GPU (CUDA:0) for inference!")
    else:
         print("⚠️ CUDA device not detected by PyTorch. Falling back to CPU.")

    # 2. Load the YOLO model
    try:
        model = YOLO(MODEL_PATH, device=DEVICE) 
    except TypeError as type_err:
        if "unexpected keyword argument 'device'" in str(type_err):
            print("⚠️ Detected an older Ultralytics version. Retrying model load without explicit 'device' argument.")
            model = YOLO(MODEL_PATH)
            DEVICE = model.device 
            print(f"✅ Model loaded successfully on auto-detected device: {DEVICE}")
        else:
            raise type_err 
    
except ImportError:
    print("❌ CRITICAL ERROR: PyTorch is not installed. Please install torch with CUDA support.")
    sys.exit()
except Exception as e:
    print(f"❌ CRITICAL ERROR: Failed to load model or check GPU: {e}")
    sys.exit()


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera failed to open. Check drivers/permissions.")
    sys.exit()

<<<<<<< HEAD
window_name = 'Rock Paper Scissors Go (RPSGO)'
=======
window_name = 'Rock Paper Scissors Arena (GPU Active)'
>>>>>>> d06c91da98281b1556a468bfd83832f960081a67
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

p1_score = 0
p2_score = 0
draws = 0

# Game States: 'WAITING', 'COUNTDOWN', 'PLAYING', 'SHOW_RESULT'
game_state = 'WAITING' 
round_start_time = 0.0

locked_winner: Optional[str] = None
locked_p1_box: Optional[Tuple[int, int, int, int]] = None
locked_p2_box: Optional[Tuple[int, int, int, int]] = None
locked_p1_gesture: Optional[str] = None
locked_p2_gesture: Optional[str] = None

rules = {
    'rock': 'scissors',
    'scissors': 'paper',
    'paper': 'rock'
}

print("\nControls: SPACE=Start/Next Round, R=Reset, Q=Quit\n")

# ============== GAME LOOP ==============
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        mid = w // 2
        
        # Resize for AI processing
        small_frame = cv2.resize(frame, (INFERENCE_W, INFERENCE_H))
        
        # Run inference
        results = model(small_frame, conf=CONFIDENCE, verbose=False, device=DEVICE) 
        
        scale_x = w / INFERENCE_W
        scale_y = h / INFERENCE_H
        
        current_p1_box = None
        current_p2_box = None
        current_p1_gesture = None
        current_p2_gesture = None

        # Process Results
        for result in results:
            for box in result.boxes:
                x1_s, y1_s, x2_s, y2_s = map(int, box.xyxy[0].cpu().numpy())
                
                # Scale up coordinates for drawing on the large frame
                x1, y1 = int(x1_s * scale_x), int(y1_s * scale_y)
                x2, y2 = int(x2_s * scale_x), int(y2_s * scale_y)

                cls = int(box.cls[0])
                gesture = result.names[cls].lower()
                center_x = (x1 + x2) / 2 
                
                # Hand Assignment Logic (Split Screen)
                if center_x < mid:
                    current_p1_gesture = gesture
                    current_p1_box = (x1, y1, x2, y2)
                elif center_x >= mid:
                    current_p2_gesture = gesture
                    current_p2_box = (x1, y1, x2, y2)

        # 3. GAME LOGIC
        if game_state == 'COUNTDOWN':
            elapsed_time = time.time() - round_start_time
            if elapsed_time >= COUNTDOWN_TIME:
                 game_state = 'PLAYING'
            
            time_left = COUNTDOWN_TIME - elapsed_time
            if time_left > 0:
                countdown_text = f"{time_left:.1f}"
            else:
                countdown_text = "GO!"
            
            cv2.putText(frame, countdown_text, (w//2 - 100, h//2 + 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255) if time_left > 0.5 else (0, 255, 0), 8)


        elif game_state == 'PLAYING':
            # Lock the gestures on the first frame both are detected
            if current_p1_gesture and current_p2_gesture:
                locked_p1_gesture = current_p1_gesture
                locked_p2_gesture = current_p2_gesture
                locked_p1_box = current_p1_box
                locked_p2_box = current_p2_box

                # Determine Winner
                if locked_p1_gesture == locked_p2_gesture:
                    locked_winner = 'draw'
                    draws += 1
                elif rules.get(locked_p1_gesture) == locked_p2_gesture:
                    locked_winner = 'p1'
                    p1_score += 1
                else:
                    locked_winner = 'p2'
                    p2_score += 1
                
                print(f"Result: {locked_p1_gesture} vs {locked_p2_gesture} -> {locked_winner.upper()}")
                game_state = 'SHOW_RESULT'
                 
        
        elif game_state == 'WAITING':
             # Clear locked results for next round
             locked_winner = locked_p1_box = locked_p2_box = None
             locked_p1_gesture = locked_p2_gesture = None
             
        # 4. DRAWING
        
        # Draw Split Line & Labels
        cv2.line(frame, (mid, 0), (mid, h), (100, 100, 100), 2)
<<<<<<< HEAD
        #cv2.putText(frame, "PLAYER 1", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        #cv2.putText(frame, "PLAYER 2", (mid + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
=======
        cv2.putText(frame, "PLAYER 1", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "PLAYER 2", (mid + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
>>>>>>> d06c91da98281b1556a468bfd83832f960081a67

        # Determine which boxes/gestures to draw (live or locked)
        box_p1 = current_p1_box
        box_p2 = current_p2_box
        gesture_p1 = current_p1_gesture
        gesture_p2 = current_p2_gesture
        
        if game_state == 'SHOW_RESULT':
            box_p1 = locked_p1_box
            box_p2 = locked_p2_box
            gesture_p1 = locked_p1_gesture
            gesture_p2 = locked_p2_gesture
        
        # Helper function to draw a box
        def draw_box(frame, box, gesture, player_winner, is_p1):
            if not box or not gesture: return
            x1, y1, x2, y2 = box
            
            # Use gesture color for live feedback
            color: Tuple[int, int, int] = GESTURE_COLORS.get(gesture, (255, 255, 255))
            label_suffix = ""

            # OVERRIDE color if showing results
            if game_state == 'SHOW_RESULT':
                if player_winner == 'draw':
                    color = (0, 255, 255) # Yellow for draw
                    label_suffix = " (DRAW)"
                elif (is_p1 and player_winner == 'p1') or (not is_p1 and player_winner == 'p2'):
                    color = (0, 255, 0) # Green for win
<<<<<<< HEAD
                    label_suffix = " (WON)"
                else:
                    color = (0, 0, 255) # Red for loss
                    label_suffix = " (LOST)"
=======
                    label_suffix = " (WINNER)"
                else:
                    color = (0, 0, 255) # Red for loss
                    label_suffix = " (LOSER)"
>>>>>>> d06c91da98281b1556a468bfd83832f960081a67
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            label = f"{'P1' if is_p1 else 'P2'}: {gesture.upper()}{label_suffix}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        draw_box(frame, box_p1, gesture_p1, locked_winner, True)
        draw_box(frame, box_p2, gesture_p2, locked_winner, False)

        # 5. DRAW UI MESSAGES
        
        if game_state == 'SHOW_RESULT':
            cv2.putText(frame, "Press SPACE for next round", (w//2 - 180, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        elif game_state == 'PLAYING':
<<<<<<< HEAD
            msg = "SHOW HANDS NOW! (move hands to fit in frame!)"
            cv2.putText(frame, msg, (w//2 - 300, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
=======
            msg = "SHOW HANDS NOW! (Locking soon...)"
            cv2.putText(frame, msg, (w//2 - 250, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
>>>>>>> d06c91da98281b1556a468bfd83832f960081a67
            
        elif game_state == 'COUNTDOWN':
             msg = "Get ready to show your hands!"
             cv2.putText(frame, msg, (w//2 - 250, h//2 - 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


        elif game_state == 'WAITING':
            msg = "Ready. Press SPACE to Start Round."
            cv2.putText(frame, msg, (w//2 - 250, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        
        # 6. DRAW SCOREBOARD
        score_board = f"{name_1}: {p1_score}  |  Draws: {draws}  |  {name_2}: {p2_score}"
        cv2.rectangle(frame, (w//2 - 200, 0), (w//2 + 200, 40), (0,0,0), -1) 
        cv2.putText(frame, score_board, (w//2 - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # DRAW QUIT INSTRUCTION
        quit_msg = "Q to Quit | R to Reset Scores"
        cv2.putText(frame, quit_msg, (w - 300, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        # DISPLAY FRAME
        cv2.imshow(window_name, frame)

        # CONTROLS
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            if game_state == 'WAITING' or game_state == 'SHOW_RESULT':
                game_state = 'COUNTDOWN'
                round_start_time = time.time() 
                print("🏁 Countdown Started!")

        elif key == ord('r'):
            p1_score = p2_score = draws = 0
            game_state = 'WAITING'
            print("Scores Reset")

except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("\nGame closed.")
