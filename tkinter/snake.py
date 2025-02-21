import pickle
import cv2
import mediapipe as mp
import numpy as np
import pygame
import sys

pygame.init()

WIDTH, HEIGHT = 640, 480
CELL_SIZE = 20
FPS = 10

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Змейка с управлением жестами")

model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
labels_dict = {0: 'Left', 1: 'Right', 2: 'One', 3: 'Two', 4: 'Three'}

camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cap = cv2.VideoCapture(1)

class Snake:
    def __init__(self):
        self.body = [(100, 100), (80, 100), (60, 100)]
        self.direction = (CELL_SIZE, 0)

    def move(self):
        new_head = (self.body[0][0] + self.direction[0], self.body[0][1] + self.direction[1])
        self.body.insert(0, new_head)
        self.body.pop()

    def grow(self):
        self.body.append(self.body[-1])

    def draw(self, screen):
        for segment in self.body:
            pygame.draw.rect(screen, GREEN, (*segment, CELL_SIZE, CELL_SIZE))

    def check_collision(self):
        head = self.body[0]
        if head[0] < 0 or head[0] >= WIDTH or head[1] < 0 or head[1] >= HEIGHT:
            return True
        if head in self.body[1:]:
            return True
        return False

class Food:
    def __init__(self):
        self.position = (0, 0)
        self.randomize_position()

    def randomize_position(self):
        self.position = (np.random.randint(0, WIDTH // CELL_SIZE) * CELL_SIZE,
                         np.random.randint(0, HEIGHT // CELL_SIZE) * CELL_SIZE)

    def draw(self, screen):
        pygame.draw.rect(screen, RED, (*self.position, CELL_SIZE, CELL_SIZE))

snake = Snake()
food = Food()

clock = pygame.time.Clock()
running = True
while running:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]
            min_x, min_y = min(x_), min(y_)
            data_aux = [lm.x - min_x for lm in hand_landmarks.landmark] + [lm.y - min_y for lm in hand_landmarks.landmark]
            
            if len(data_aux) == 42:
                prediction = model.predict([np.asarray(data_aux)])
                gesture = labels_dict.get(int(prediction[0]), None)
                
                if gesture == 'Left':
                    if snake.direction != (CELL_SIZE, 0):
                        snake.direction = (-CELL_SIZE, 0)
                elif gesture == 'Right':
                    if snake.direction != (-CELL_SIZE, 0):
                        snake.direction = (CELL_SIZE, 0)
                elif gesture == 'One':
                    if snake.direction != (0, CELL_SIZE):
                        snake.direction = (0, -CELL_SIZE)
                elif gesture == 'Two':
                    if snake.direction != (0, -CELL_SIZE):
                        snake.direction = (0, CELL_SIZE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    snake.move()

    if snake.body[0] == food.position:
        snake.grow()
        food.randomize_position()

    if snake.check_collision():
        running = False

    screen.fill(BLACK)
    snake.draw(screen)
    food.draw(screen)
    pygame.display.flip()

    clock.tick(FPS)

cap.release()
cv2.destroyAllWindows()
pygame.quit()
