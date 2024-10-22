#!/usr/bin/env python3
# Based on https://python101.readthedocs.io/pl/latest/pygame/pong/#
import pygame
from typing import Type
import skfuzzy as fuzz
import skfuzzy.control as fuzzcontrol

FPS = 30


class Board:
    def __init__(self, width: int, height: int):
        self.surface = pygame.display.set_mode((width, height), 0, 32)
        pygame.display.set_caption("AIFundamentals - PongGame")

    def draw(self, *args):
        background = (0, 0, 0)
        self.surface.fill(background)
        for drawable in args:
            drawable.draw_on(self.surface)

        pygame.display.update()


class Drawable:
    def __init__(self, x: int, y: int, width: int, height: int, color=(255, 255, 255)):
        self.width = width
        self.height = height
        self.color = color
        self.surface = pygame.Surface(
            [width, height], pygame.SRCALPHA, 32
        ).convert_alpha()
        self.rect = self.surface.get_rect(x=x, y=y)

    def draw_on(self, surface):
        surface.blit(self.surface, self.rect)


class Ball(Drawable):
    def __init__(
        self,
        x: int,
        y: int,
        radius: int = 20,
        color=(255, 10, 0),
        speed: int = 3,
    ):
        super(Ball, self).__init__(x, y, radius, radius, color)
        pygame.draw.ellipse(self.surface, self.color, [0, 0, self.width, self.height])
        self.x_speed = speed
        self.y_speed = speed
        self.start_speed = speed
        self.start_x = x
        self.start_y = y
        self.start_color = color
        self.last_collision = 0

    def bounce_y(self):
        self.y_speed *= -1

    def bounce_x(self):
        self.x_speed *= -1

    def bounce_y_power(self):
        self.color = (
            self.color[0],
            self.color[1] + 10 if self.color[1] < 255 else self.color[1],
            self.color[2],
        )
        pygame.draw.ellipse(self.surface, self.color, [0, 0, self.width, self.height])
        self.x_speed *= 1.1
        self.y_speed *= 1.1
        self.bounce_y()

    def reset(self):
        self.rect.x = self.start_x
        self.rect.y = self.start_y
        self.x_speed = self.start_speed
        self.y_speed = self.start_speed
        self.color = self.start_color
        self.bounce_y()

    def move(self, board: Board, *args):
        self.rect.x += round(self.x_speed)
        self.rect.y += round(self.y_speed)

        if self.rect.x < 0 or self.rect.x > (
            board.surface.get_width() - self.rect.width
        ):
            self.bounce_x()

        if self.rect.y < 0 or self.rect.y > (
            board.surface.get_height() - self.rect.height
        ):
            self.reset()

        timestamp = pygame.time.get_ticks()
        if timestamp - self.last_collision < FPS * 4:
            return

        for racket in args:
            if self.rect.colliderect(racket.rect):
                self.last_collision = pygame.time.get_ticks()
                if (self.rect.right < racket.rect.left + racket.rect.width // 4) or (
                    self.rect.left > racket.rect.right - racket.rect.width // 4
                ):
                    self.bounce_y_power()
                else:
                    self.bounce_y()


class Racket(Drawable):
    def __init__(
        self,
        x: int,
        y: int,
        width: int = 80,
        height: int = 20,
        color=(255, 255, 255),
        max_speed: int = 10,
    ):
        super(Racket, self).__init__(x, y, width, height, color)
        self.max_speed = max_speed
        self.surface.fill(color)

    def move(self, x: int, board: Board):
        delta = x - self.rect.x
        delta = self.max_speed if delta > self.max_speed else delta
        delta = -self.max_speed if delta < -self.max_speed else delta
        delta = 0 if (self.rect.x + delta) < 0 else delta
        delta = (
            0
            if (self.rect.x + self.width + delta) > board.surface.get_width()
            else delta
        )
        self.rect.x += delta


class Player:
    def __init__(self, racket: Racket, ball: Ball, board: Board) -> None:
        self.ball = ball
        self.racket = racket
        self.board = board

    def move(self, x: int):
        self.racket.move(x, self.board)

    def move_manual(self, x: int):
        """
        Do nothing, control is defined in derived classes
        """
        pass

    def act(self, x_diff: int, y_diff: int):
        """
        Do nothing, control is defined in derived classes
        """
        pass


class PongGame:
    def __init__(
        self, width: int, height: int, player1: Type[Player], player2: Type[Player]
    ):
        pygame.init()
        self.board = Board(width, height)
        self.fps_clock = pygame.time.Clock()
        self.ball = Ball(width // 2, height // 2)

        self.opponent_paddle = Racket(x=width // 2, y=0)
        self.oponent = player1(self.opponent_paddle, self.ball, self.board)

        self.player_paddle = Racket(x=width // 2, y=height - 20)
        self.player = player2(self.player_paddle, self.ball, self.board)

    def run(self):
        while not self.handle_events():
            self.ball.move(self.board, self.player_paddle, self.opponent_paddle)
            self.board.draw(
                self.ball,
                self.player_paddle,
                self.opponent_paddle,
            )
            self.oponent.act(
                self.oponent.racket.rect.centerx - self.ball.rect.centerx,
                self.oponent.racket.rect.centery - self.ball.rect.centery,
            )
            self.player.act(
                self.player.racket.rect.centerx - self.ball.rect.centerx,
                self.player.racket.rect.centery - self.ball.rect.centery,
            )
            self.fps_clock.tick(FPS)

    def handle_events(self):
        for event in pygame.event.get():
            if (event.type == pygame.QUIT) or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                pygame.quit()
                return True
        keys = pygame.key.get_pressed()
        if keys[pygame.constants.K_LEFT]:
            self.player.move_manual(0)
        elif keys[pygame.constants.K_RIGHT]:
            self.player.move_manual(self.board.surface.get_width())
        return False


class NaiveOponent(Player):
    def __init__(self, racket: Racket, ball: Ball, board: Board):
        super(NaiveOponent, self).__init__(racket, ball, board)

    def act(self, x_diff: int, y_diff: int):
        x_cent = self.ball.rect.centerx
        self.move(x_cent)


class HumanPlayer(Player):
    def __init__(self, racket: Racket, ball: Ball, board: Board):
        super(HumanPlayer, self).__init__(racket, ball, board)

    def move_manual(self, x: int):
        self.move(x)


# ----------------------------------
# DO NOT MODIFY CODE ABOVE THIS LINE
# ----------------------------------

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzzy

class FuzzyPlayer(Player):
    def __init__(self, racket: Racket, ball: Ball, board: Board):
        super(FuzzyPlayer, self).__init__(racket, ball, board)

        print(racket)
        print(ball.height)
        # for Mamdami:
        x_dist = fuzz.control.Antecedent(np.arange(-800,800,10),'x_dist')
        y_dist = fuzz.control.Antecedent(np.arange(0,400,10),'y_dist')
        velocity = fuzz.control.Consequent(np.arange(-20,20,1),'velocity')
        x_dist['far_left'] = fuzz.trimf(x_dist.universe,[-800,-800,-400])
        x_dist['left'] = fuzz.trimf(x_dist.universe,[-600,-300,0])
        x_dist['center'] = fuzz.trimf(x_dist.universe,[-20,0,20])
        x_dist['right'] = fuzz.trimf(x_dist.universe,[0,300,600])
        x_dist['far_right'] = fuzz.trimf(x_dist.universe,[400,800,800])

        y_dist['far'] = fuzz.trimf(y_dist.universe,[200,400,400])
        y_dist['near'] = fuzz.trimf(y_dist.universe,[0,200,400])
        y_dist['very_near'] = fuzz.trimf(y_dist.universe,[0,0,200])

        velocity['very_fast_right'] = fuzz.trimf(velocity.universe,[-20,-20,-15])
        velocity['fast_right'] = fuzz.trimf(velocity.universe,[-20,-15,-10])
        velocity['right'] = fuzz.trimf(velocity.universe,[-15,-10,0])
        velocity['stop'] = fuzz.trimf(velocity.universe,[-10,0,10])
        velocity['left'] = fuzz.trimf(velocity.universe,[0,10,15])
        velocity['fast_left'] = fuzz.trimf(velocity.universe,[10,15,20])
        velocity['very_fast_left'] = fuzz.trimf(velocity.universe,[15,20,20])

        x_dist.view()
        y_dist.view()
        velocity.view()

        plt.show()


        rules = [
            fuzz.control.Rule(x_dist['far_left'] & y_dist["near"] ,velocity['very_fast_left']),
            fuzz.control.Rule(x_dist['far_left'] & y_dist["far"] ,velocity['fast_left']),
            fuzz.control.Rule(x_dist['left'] & y_dist["very_near"],velocity['very_fast_left']),
            fuzz.control.Rule(x_dist['left'] & y_dist["near"],velocity['fast_left']),
            fuzz.control.Rule(x_dist['left'] & y_dist["far"],velocity['left']),
            fuzz.control.Rule(x_dist['center'] ,velocity['stop']),
            fuzz.control.Rule(x_dist['right'] & y_dist["far"],velocity['right']),
            fuzz.control.Rule(x_dist['right'] & y_dist["near"],velocity['fast_right']),
            fuzz.control.Rule(x_dist['right'] & y_dist["very_near"],velocity['very_fast_right']),
            fuzz.control.Rule(x_dist['far_right'] & y_dist["far"] ,velocity['fast_right']),
            fuzz.control.Rule(x_dist['far_right'] & y_dist["near"],velocity['very_fast_right']),
        ]

        idk = fuzz.control.ControlSystem(rules)

        self.racket_controller = fuzz.control.ControlSystemSimulation(idk)




    def act(self, x_diff: int, y_diff: int):
        velocity = self.make_decision(x_diff, y_diff)
        self.move(self.racket.rect.x + velocity)

    def make_decision(self, x_diff: int, y_diff: int):

        print(x_diff,y_diff)
        # for Mamdami:
            #
        self.racket_controller.input['x_dist'] = x_diff
        self.racket_controller.input['y_dist'] = y_diff
        self.racket_controller.compute()

        print(self.racket_controller.output)
        return self.racket_controller.output['velocity']


class TskFuzzyPlayer(Player):
    def __init__(self, racket: Racket, ball: Ball, board: Board):
        super(TskFuzzyPlayer, self).__init__(racket, ball, board)


        # for TSK:
        self.x_universe = np.arange(-800,800,10)
        self.y_universe = np.arange(0,400,10)
        self.x_mf = {
            "far_left": fuzz.trapmf(self.x_universe,[-800,-800,-600,-200]),
            "left": fuzz.trapmf(self.x_universe,[-400,-200,-100,100]),
            "right": fuzz.trapmf(self.x_universe,[-100,100,200,400]),
            "far_right": fuzz.trapmf(self.x_universe,[200,600,800,800]),
        }

        self.y_mf = {
            "near" : fuzz.trapmf(self.y_universe,[0,0,100,200]),
            "far" :fuzz.trapmf(self.y_universe,[100,200,400,400])
        }

        self.velocity_fx = {
            "f_slow_left": lambda x_diff, y_diff: 1 * (abs(x_diff) + y_diff),
            "f_fast_left": lambda x_diff, y_diff: 2 * (abs(x_diff) + y_diff),
            "f_slow_right": lambda x_diff, y_diff: -1 * (abs(x_diff) + y_diff),
            "f_fast_right": lambda x_diff, y_diff: -2 * (abs(x_diff) + y_diff),
        }

        # visualize TSK
        plt.figure()
        for name, mf in self.x_mf.items():
            plt.plot(self.x_universe, mf, label=name)
        plt.legend()
        plt.show()
        # ...

    def act(self, x_diff: int, y_diff: int):
        velocity = self.make_decision(x_diff, y_diff)
        self.move(self.racket.rect.x + velocity)

    def make_decision(self, x_diff: int, y_diff: int):

        print(x_diff,y_diff)
        # for TSK:
        x_vals = {
            name: fuzz.interp_membership(self.x_universe, mf, x_diff)
            for name, mf in self.x_mf.items()
        }

        y_vals = {
            name: fuzz.interp_membership(self.y_universe, mf, y_diff)
            for name, mf in self.y_mf.items()
        }
        print(x_vals)
        # ...
        # rule activations with Zadeh norms
        activations = {
            "f_slow_left": max(
                [
                    min(x_vals["far_left"],y_vals["far"]),
                    min(x_vals["left"],y_vals["near"]),
                    min(x_vals["left"],y_vals["far"]),
                ]
            ),
            "f_slow_right": max(
                [
                    min(x_vals["far_right"],y_vals["far"]),
                    min(x_vals["right"],y_vals["near"]),
                    min(x_vals["right"],y_vals["far"]),
                ]
            ),
            "f_fast_right": max(
                [
                    min(x_vals["far_right"],y_vals["near"]),
                ]
            ),
            "f_fast_left": max(
                [
                    min(x_vals["far_left"],y_vals["near"]),
                ]
            )
        }


        velocity = sum(
            activations[val] * self.velocity_fx[val](x_diff, y_diff)
            for val in activations
        ) / sum(activations[val] for val in activations)
        return velocity

class EdgeFuzzyPlayer(Player):
    def __init__(self, racket: Racket, ball: Ball, board: Board):
        super(EdgeFuzzyPlayer, self).__init__(racket, ball, board)


        # for TSK:
        self.x_universe = np.arange(-800,800,10)
        self.y_universe = np.arange(0,400,10)


        self.x_mf = {
            "far_left": fuzz.trapmf(self.x_universe,[-800,-800,-600,-200]),
            "left": fuzz.trapmf(self.x_universe,[-400,-200,-40,-40]),
            "edge_left": fuzz.trapmf(self.x_universe,[-40,-40,-20,-20]),
            "center": fuzz.trapmf(self.x_universe,[-20,-20,20,20]),
            "edge_right": fuzz.trapmf(self.x_universe,[20,20,40,40]),
            "right": fuzz.trapmf(self.x_universe,[40,40,200,400]),
            "far_right": fuzz.trapmf(self.x_universe,[200,600,800,800]),
        }

        self.y_mf = {
            "near" : fuzz.trapmf(self.y_universe,[0,0,100,200]),
            "far" :fuzz.trapmf(self.y_universe,[100,200,400,400])
        }

        self.velocity_fx = {
            "f_slow_left": lambda x_diff, y_diff: 1 * (abs(x_diff) + y_diff),
            "f_fast_left": lambda x_diff, y_diff: 2 * (abs(x_diff) + y_diff),
            "stop" : lambda _,_x : 0,
            "f_slow_right": lambda x_diff, y_diff: -1 * (abs(x_diff) + y_diff),
            "f_fast_right": lambda x_diff, y_diff: -2 * (abs(x_diff) + y_diff),
        }

        # visualize TSK
        plt.figure()
        for name, mf in self.x_mf.items():
            plt.plot(self.x_universe, mf, label=name)
        plt.legend()
        plt.show()
        # ...

    def act(self, x_diff: int, y_diff: int):
        velocity = self.make_decision(x_diff, y_diff)
        self.move(self.racket.rect.x + velocity)

    def make_decision(self, x_diff: int, y_diff: int):

        print(x_diff,y_diff)
        # for TSK:
        x_vals = {
            name: fuzz.interp_membership(self.x_universe, mf, x_diff)
            for name, mf in self.x_mf.items()
        }

        y_vals = {
            name: fuzz.interp_membership(self.y_universe, mf, y_diff)
            for name, mf in self.y_mf.items()
        }
        print(x_vals)
        # ...
        # rule activations with Zadeh norms
        activations = {
            "f_slow_left": max(
                [
                    min(x_vals["far_left"],y_vals["far"]),
                    min(x_vals["left"],y_vals["near"]),
                    min(x_vals["left"],y_vals["far"]),
                ]
            ),
            "f_slow_right": max(
                [
                    min(x_vals["far_right"],y_vals["far"]),
                    min(x_vals["right"],y_vals["near"]),
                    min(x_vals["right"],y_vals["far"]),
                ]
            ),
            "f_fast_right": max(
                [
                    min(x_vals["far_right"],y_vals["near"]),
                ]
            ),
            "f_fast_left": max(
                [
                    min(x_vals["far_left"],y_vals["near"]),
                    min(x_vals["center"],1)
                ]
            ),
            "stop" : max(
                [
                    min(x_vals["edge_left"],1),
                    min(x_vals["edge_right"],1)
                ]
            )
        }


        velocity = sum(
            activations[val] * self.velocity_fx[val](x_diff, y_diff)
            for val in activations
        ) / sum(activations[val] for val in activations)
        return velocity


if __name__ == "__main__":
    #game = PongGame(800, 400, NaiveOponent, HumanPlayer)
    game = PongGame(800, 400, NaiveOponent, FuzzyPlayer)
    # game = PongGame(800, 400, NaiveOponent, TskFuzzyPlayer)

    # game = PongGame(800, 400, NaiveOponent, EdgeFuzzyPlayer)
    game.run()
