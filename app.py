import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

# Constants
WINDOWSIZEX = 640
WINDOWSIZEY = 480
BOUNDRYINC = 5

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

IMAGESAVE = False
PREDICT = True

# Load model
model = load_model('best_model.h5')

# Labels
LABELS = {
    0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four",
    5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"
}

# Initialize pygame
pygame.init()
FONT = pygame.font.SysFont("freesansbold.ttf", 20)
BIGFONT = pygame.font.SysFont("freesansbold.ttf", 28)

DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption('Digit Recognition')
DISPLAYSURF.fill(BLACK)

# Title text
title_text = BIGFONT.render("Handwritten Digit Recognition", True, RED)
title_rect = title_text.get_rect(center=(WINDOWSIZEX // 2, 25))
DISPLAYSURF.blit(title_text, title_rect)

# Clear Button
clear_button = pygame.Rect(WINDOWSIZEX - 110, 10, 90, 30)
pygame.draw.rect(DISPLAYSURF, WHITE, clear_button)
clear_text = FONT.render("Clear", True, BLACK)
DISPLAYSURF.blit(clear_text, (WINDOWSIZEX - 90, 15))

# Variables
iswriting = False
number_xcord = []
number_ycord = []
image_cnt = 1

# Main loop
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            if ycord > 50:  # prevent drawing over title area
                pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)
                number_xcord.append(xcord)
                number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            x, y = event.pos
            # Check if clear button clicked
            if clear_button.collidepoint(x, y):
                DISPLAYSURF.fill(BLACK)
                DISPLAYSURF.blit(title_text, title_rect)
                pygame.draw.rect(DISPLAYSURF, WHITE, clear_button)
                DISPLAYSURF.blit(clear_text, (WINDOWSIZEX - 90, 15))
                continue
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            if number_xcord and number_ycord:
                number_xcord = sorted(number_xcord)
                number_ycord = sorted(number_ycord)

                rect_min_x = max(number_xcord[0] - BOUNDRYINC, 0)
                rect_max_x = min(number_xcord[-1] + BOUNDRYINC, WINDOWSIZEX)
                rect_min_y = max(number_ycord[0] - BOUNDRYINC, 0)
                rect_max_y = min(number_ycord[-1] + BOUNDRYINC, WINDOWSIZEY)

                number_xcord = []
                number_ycord = []

                img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

                if IMAGESAVE:
                    cv2.imwrite(f"image_{image_cnt}.png", img_arr)
                    image_cnt += 1

                if PREDICT:
                    image = cv2.resize(img_arr, (28, 28))
                    image = np.pad(image, (10, 10), 'constant', constant_values=0)
                    image = cv2.resize(image, (28, 28)) / 255.0
                    image = image.reshape(1, 28, 28, 1)

                    prediction = model.predict(image)
                    label = str(LABELS[np.argmax(prediction)])

                    pygame.draw.rect(DISPLAYSURF, RED, (rect_min_x, rect_min_y, rect_max_x - rect_min_x, rect_max_y - rect_min_y), 2)

                    textsurface = FONT.render(label, True, RED, WHITE)
                    textRecObj = textsurface.get_rect()
                    textRecObj.left, textRecObj.top = rect_min_x, rect_min_y - 20
                    DISPLAYSURF.blit(textsurface, textRecObj)

        if event.type == KEYDOWN:
            if event.unicode.lower() == "n":
                DISPLAYSURF.fill(BLACK)
                DISPLAYSURF.blit(title_text, title_rect)
                pygame.draw.rect(DISPLAYSURF, WHITE, clear_button)
                DISPLAYSURF.blit(clear_text, (WINDOWSIZEX - 90, 15))

    pygame.display.update()
