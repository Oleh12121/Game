import time
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from enum import Enum
from collections import namedtuple, deque
import numpy as np
import random
import argparse
from IPython import display
import pygame
import sys
import os

pygame.init()
font = pygame.font.Font(None, 25)  
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLOCK_SIZE = 20
SPEED = 40
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.1
GAMMA = 0.9
WIDTH = 900
HEIGHT = 600
WHITE = (255, 255, 255)
BUTTON_SIZE = 150
BUTTON_RADIUS = 30  
BUTTON_SPACING = 100  
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BOARD_SIZE = (6, 7)

class c4OverflowRune(Exception):
    def __init__(self, omen):
        self.omen = omen
    def __str__(self):
        return repr(self.omen)

class c4HexCell:
    RUNIC_CAPACITY = 80

    def __init__(self, glyph_r, glyph_c, dim_w, dim_h, pos_x, pos_y):
        self.payload = 0
        self.glyph_r = glyph_r
        self.glyph_c = glyph_c
        self.dim_w = dim_w
        self.dim_h = dim_h
        self.surface = pygame.Surface((dim_w*2, dim_h*2))
        self.coord_x = pos_x
        self.coord_y = pos_y

    def reveal_coordinates(self):
        return (self.glyph_r, self.glyph_c)

    def pinpoint_reference(self):
        return (self.coord_x, self.coord_y)

    def inscribe_token(self, orb):
        self.payload = orb.reveal_type()

    def detect_occupancy(self):
        return self.payload != 0

    def retrieve_payload(self):
        return self.payload

    def render_cell(self, canvas):
        pygame.draw.rect(self.surface, BLACK, (0, 0, self.dim_w, self.dim_h))
        pygame.draw.rect(self.surface, WHITE, (0, 0, self.dim_w - 2, self.dim_h - 2))
        self.surface = self.surface.convert()
        canvas.blit(self.surface, (self.coord_x, self.coord_y))

class c4GridSchema:
    GRID_OFFSET_X = 150
    GRID_OFFSET_Y = 100

    def __init__(self, rows, cols):
        self.chart = [[c4HexCell(r, c, c4HexCell.RUNIC_CAPACITY, c4HexCell.RUNIC_CAPACITY,
                        c* c4HexCell.RUNIC_CAPACITY + c4GridSchema.GRID_OFFSET_X,
                        r* c4HexCell.RUNIC_CAPACITY + c4GridSchema.GRID_OFFSET_Y)
                       for c in range(cols)] for r in range(rows)]
        self.rows = rows
        self.cols = cols
        self.total_cells = rows * cols
        self.filled_count = 0
        self.recent_path = []
        self.last_mark = 0
        self.snapshot = [[0]*cols for _ in range(rows)]
        self.prev_snapshot = None
        self.prev_mark = (None, None, None)
        self.network = [[c4NodeGlyph() for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                node = self.network[r][c]
                if r>0 and c>0: node.top_left = self.network[r-1][c-1]
                if r>0: node.top = self.network[r-1][c]
                if r>0 and c<cols-1: node.top_right = self.network[r-1][c+1]
                if c>0: node.left = self.network[r][c-1]
                if c<cols-1: node.right = self.network[r][c+1]
                if r<rows-1 and c>0: node.bottom_left = self.network[r+1][c-1]
                if r<rows-1: node.bottom = self.network[r+1][c]
                if r<rows-1 and c<cols-1: node.bottom_right = self.network[r+1][c+1]

    def render_schema(self, canvas):
        for r in range(self.rows):
            for c in range(self.cols):
                self.chart[r][c].render_cell(canvas)

    def fetch_cell(self, glyph_r, glyph_c):
        return self.chart[glyph_r][glyph_c]

    def assess_column_overflow(self, c):
        return all(self.chart[r][c].detect_occupancy() for r in range(self.rows))

    def engrave_disc(self, orb, canvas, augur):
        c = orb.reveal_column()
        if not self.assess_column_overflow(c):
            r = self.compute_drop_row(c)
            self.chart[r][c].inscribe_token(orb)
            if self.prev_mark[0] is None:
                self.prev_snapshot = [[0]*self.cols for _ in range(self.rows)]
            else:
                pr, pc, val = self.prev_mark
                self.prev_snapshot[pr][pc] = val
            self.prev_mark = (r, c, orb.reveal_type())
            self.snapshot[r][c] = orb.reveal_type()
            self.rune_traversal_init(r, c, orb.reveal_type())
            self.filled_count += 1
            self.last_mark = orb.reveal_type()
            orb.plummet_disc(canvas, r)
        else:
            raise c4OverflowRune('Column brimming with fate!')
        return augur.assess_epoch()

    def compute_drop_row(self, c):
        for r in range(self.rows):
            if self.chart[r][c].detect_occupancy():
                return r-1
        return self.rows-1

    def schema_dimensions(self):
        return (self.rows, self.cols)

    def verify_fullness(self):
        return self.total_cells == self.filled_count

    def glyph_network(self):
        return self.network

    def enumerate_moves(self):
        return [i for i in range(self.cols) if not self.assess_column_overflow(i)]

    def snapshot_current(self):
        return tuple(tuple(row) for row in self.snapshot)

    def snapshot_previous(self):
        return tuple(tuple(row) for row in self.prev_snapshot)

    def retrieve_recent(self):
        return (self.recent_path, self.last_mark)

    def rune_traversal_init(self, r, c, mark):
        self.recent_path = []
        start = self.network[r][c]
        start.value = mark
        self.rune_traverse(start, mark, r, c, self.recent_path)
        for (rr, cc) in self.recent_path:
            self.network[rr][cc].visited = False

    def rune_traverse(self, node, mark, r, c, trail):
        node.visited = True
        trail.append((r, c))
        dirs = [
            ('top_left', -1, -1), ('top', -1, 0), ('top_right', -1, 1),
            ('left', 0, -1), ('right', 0, 1),
            ('bottom_left', 1, -1), ('bottom', 1, 0), ('bottom_right', 1, 1)
        ]
        for attr, dr, dc in dirs:
            nbr = getattr(node, attr)
            if nbr and nbr.value == mark:
                setattr(node, f"{attr}_score", getattr(nbr, f"{attr}_score") + 1)
                if not nbr.visited:
                    self.rune_traverse(nbr, mark, r+dr, c+dc, trail)

class c4VistaEngine:
    def __init__(self, wid=900, hgt=600, tempo=30):
        pygame.init()
        pygame.display.set_caption("З'єднай 4")
        self.wid = wid
        self.hgt = hgt
        self.canvas = pygame.display.set_mode((wid, hgt), pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()
        self.tempo = tempo
        self.playtime = 0.0
        self.font = pygame.font.SysFont('mono', 20, bold=True)
        self.oracle = None
        self.victory_tallies = [0,0,0]
        self.score_log = []
        self.mean_log = []
        plt.ion()

    def draft_insights(self, logs, means, ties):
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Number of Episodes')
        plt.ylabel('Cumulative Wins')
        plt.plot(logs, label='Agent1 wins')
        plt.plot(means, label='Agent2 wins')
        plt.plot(ties, label='Draws')
        plt.ylim(ymin=0)
        plt.legend()
        plt.pause(.1)

    def configure_encounter(self, mode):
        self.schema = c4GridSchema(BOARD_SIZE[0], BOARD_SIZE[1])
        self.dim_r, self.dim_c = self.schema.schema_dimensions()
        self.augur = c4OracleMatrix(self.schema)
        first_mark = random.randint(1,2)
        second_mark = 2 if first_mark==1 else 1
        if mode=='solo':
            self.agent1 = c4AgentTactician(first_mark)
            if self.oracle is None:
                self.agent2 = c4AgentOracle(second_mark, 'qlearner')
                self.oracle = self.agent2
            else:
                self.oracle.calibrate_type(second_mark)
                self.agent2 = self.oracle

    def configure_scholarship(self):
        self.schema = c4GridSchema(BOARD_SIZE[0], BOARD_SIZE[1])
        self.dim_r, self.dim_c = self.schema.schema_dimensions()
        self.augur = c4OracleMatrix(self.schema)
        self.oracle = None
        self.victory_tallies = [0,0,0]
        self.agent1 = c4AgentOracle(1, 'qlearner')
        self.agent2 = c4AgentOracle(2, 'qlearner')
        self.iter_total = self.tempo
        self.log1 = []
        self.log2 = []
        self.log_draw = []
        self.cur1 = self.cur2 = self.curd = 0

    def helm_overview(self, episodes=20):
        self.tempo = episodes
        lobby = True
        commence = False
        self.canvas.fill(WHITE)
        self.sketch_portal()
        while lobby:
            for ev in pygame.event.get():
                if ev.type==pygame.MOUSEBUTTONDOWN:
                    x,y = pygame.mouse.get_pos()
                    if self.btn1.collidepoint((x,y)):
                        commence=True; lobby=False; mode='solo'
                    elif self.btn2.collidepoint((x,y)):
                        commence=True; lobby=False; mode='scholar'
                    elif self.btn3.collidepoint((x,y)):
                        lobby=False
                if ev.type==pygame.KEYDOWN and ev.key==pygame.K_ESCAPE:
                    lobby=False
            pygame.display.flip()
            self.canvas.blit(self.canvas, (0,0))
        if not commence:
            pygame.quit()
        elif mode=='scholar':
            self.embark_journey(mode, episodes)
        else:
            self.embark_journey(mode)

    def embark_journey(self, mode, episodes=1):
        self.configure_scholarship()
        count=0
        while episodes>0:
            self.configure_encounter(mode)
            self.canvas.fill(BLACK)
            self.schema.render_schema(self.canvas)
            over=False; init=True
            turn=random.randint(1,2)
            human_turn = (mode=='solo' and self.agent1.oracle_type()==turn)
            p1_turn = (self.agent1.oracle_type()==turn)
            x0,y0 = self.schema.fetch_cell(0,0).pinpoint_reference()
            orb = c4DiscToken(turn)
            while not over:
                if init:
                    orb = c4DiscToken(turn)
                    orb.anchor_coordinates(x0, y0 - c4HexCell.RUNIC_CAPACITY)
                    orb.define_column(0)
                    init=False; placed=False
                current = self.agent1 if p1_turn else self.agent2
                if human_turn:
                    for ev in pygame.event.get():
                        if ev.type==pygame.QUIT: over=True
                        elif ev.type==pygame.MOUSEMOTION:
                            mx,_ = ev.pos
                            col = int((mx - c4GridSchema.GRID_OFFSET_X)/c4HexCell.RUNIC_CAPACITY)
                            if 0<=col< self.dim_c:
                                orb.define_column(col)
                                orb.anchor_coordinates(c4GridSchema.GRID_OFFSET_X + col*c4HexCell.RUNIC_CAPACITY, c4GridSchema.GRID_OFFSET_Y - c4HexCell.RUNIC_CAPACITY)
                                self.canvas.fill(BLACK, (0,0,self.wid,c4GridSchema.GRID_OFFSET_Y))
                                orb.depict_orb(self.canvas)
                        elif ev.type==pygame.MOUSEBUTTONDOWN and not placed:
                            try:
                                over = self.schema.engrave_disc(orb, self.canvas, self.augur)
                                current.complete_move()
                                init=True; placed=True
                            except c4OverflowRune:
                                pass
                else:
                    over = current.orchestrate_move(orb, self.schema, self.augur, self.canvas)
                    placed=True; init=True
                if over:
                    victor = self.augur.query_victor()
                    if mode=='scholar':
                        count+=1
                        if victor==1: self.cur1+=1
                        elif victor==2: self.cur2+=1
                        else: self.curd+=1
                        self.log1.append(self.cur1); self.log2.append(self.cur2); self.log_draw.append(self.curd)
                        self.chart_progress()
                        self.draft_insights(self.log1, self.log2, self.log_draw)
                if placed:
                    human_turn = (not human_turn if mode=='solo' else False)
                    turn = 1 if turn==2 else 2
                    p1_turn = not p1_turn
                pygame.display.flip()
                self.canvas.blit(self.canvas,(0,0))
            episodes-=1
        if mode=='scholar':
            self.helm_overview()
        else:
            self.epoch_conclusion(self.augur.pronounce_victor())

    def sketch_portal(self):
        f = pygame.font.SysFont('mono',50,bold=True)
        title = f.render("З'ЄДНАЙ 4",True,BLACK)
        w,_ = f.size("З'ЄДНАЙ 4")
        self.canvas.blit(title, ((self.wid-w)//2,150))
        f2 = pygame.font.SysFont('mono',30,bold=True)
        t1='Грати'; t2='Тренування'; t3='Вийти'
        surf1=f2.render(t1,True,BLACK);w1,_=f2.size(t1)
        self.btn1=surf1.get_rect(topleft=((self.wid-w1)//2,300)); self.canvas.blit(surf1, self.btn1)
        surf2=f2.render(t2,True,BLACK);w2,_=f2.size(t2)
        self.btn2=surf2.get_rect(topleft=((self.wid-w2)//2,350)); self.canvas.blit(surf2,self.btn2)
        surf3=f2.render(t3,True,BLACK);w3,_=f2.size(t3)
        self.btn3=surf3.get_rect(topleft=((self.wid-w3)//2,400)); self.canvas.blit(surf3,self.btn3)

    def chart_progress(self):
        gx,gy,gw,gh = 900,150,300,480
        self.canvas.fill(BLACK,(gx,gy,gw,gh))
        pygame.draw.line(self.canvas,WHITE,(gx,gy+gh),(gx+gw,gy+gh))
        pygame.draw.line(self.canvas,WHITE,(gx,gy),(gx,gy+gh))
        if len(self.log1)>1:
            pts1=[(gx+(i/(self.iter_total-1))*gw, gy+gh-(self.log1[i]/self.iter_total)*gh) for i in range(len(self.log1))]
            pygame.draw.lines(self.canvas,BLUE,False,pts1)
            pts2=[(gx+(i/(self.iter_total-1))*gw, gy+gh-(self.log2[i]/self.iter_total)*gh) for i in range(len(self.log2))]
            pygame.draw.lines(self.canvas,RED,False,pts2)
            ptsd=[(gx+(i/(self.iter_total-1))*gw, gy+gh-(self.log_draw[i]/self.iter_total)*gh) for i in range(len(self.log_draw))]
            pygame.draw.lines(self.canvas,WHITE,False,ptsd)

    def epoch_conclusion(self, victor):
        over=True; menu=False
        self.canvas.fill(WHITE)
        self.herald_resolve(victor)
        while over:
            for ev in pygame.event.get():
                if ev.type==pygame.MOUSEBUTTONDOWN:
                    if self.btnA.collidepoint(pygame.mouse.get_pos()): menu=True; over=False
                    elif self.btnB.collidepoint(pygame.mouse.get_pos()): over=False
                if ev.type==pygame.KEYDOWN and ev.key==pygame.K_ESCAPE: over=False
            pygame.display.flip()
            self.canvas.blit(self.canvas,(0,0))
        if not menu: pygame.quit()
        else: self.helm_overview()

    def herald_resolve(self, victor):
        f=pygame.font.SysFont('mono',50,bold=True)
        txt=f.render('ГРА ЗАВЕРШЕНА',True,GREEN)
        w,_=f.size('ГРА ЗАВЕРШЕНА'); self.canvas.blit(txt,((self.wid-w)//2,150))
        f2=pygame.font.SysFont('mono',30,bold=True)        
        wt= f"{victor} виграв!" if victor != 'TIE' else 'НІЧИЯ!' ; wsurf=f2.render(wt,True,BLACK)
        w,_=f2.size(wt); self.canvas.blit(wsurf,((self.wid-w)//2,300))
        f3=pygame.font.SysFont('mono',40)
        a1=f3.render('Повернутись до головного меню',True,BLACK); w1,_=f3.size('Повернутись до головного меню'); self.btnA=a1.get_rect(topleft=((self.wid-w1)//2,360)); self.canvas.blit(a1,self.btnA)
        a2=f3.render('Вийти',True,BLACK); w2,_=f3.size('Вийти'); self.btnB=a2.get_rect(topleft=((self.wid-w2)//2,410)); self.canvas.blit(a2,self.btnB)

class c4AgentBase:
    def __init__(self, mark):
        self.mark = mark
    def complete_move(self): pass
    def oracle_type(self): return self.mark
    def calibrate_type(self, mark): self.mark = mark

class c4AgentTactician(c4AgentBase): pass

class c4AgentOracle(c4AgentBase):
    def __init__(self, mark, kind):
        super().__init__(mark)
        self.core = c4AgentScholar(mark) if kind=='qlearner' else c4AgentChaos(mark)
    def orchestrate_move(self, orb, schema, augur, canvas):
        acts = schema.enumerate_moves()
        state = schema.snapshot_current()
        choice = self.select_action(state, acts)
        orb.shift_east(canvas, choice)
        orb.define_column(choice)
        over = schema.engrave_disc(orb, canvas, augur)
        self.core.refine_rune(schema, acts, choice, over, augur)
        return over
    def select_action(self, state, acts): return self.core.select_rune(state, acts)

class c4AgentChaos(c4AgentBase):
    def select_rune(self, state, acts): return random.choice(acts)
    def refine_rune(self, *args): pass

class c4AgentScholar(c4AgentBase):
    def __init__(self, mark, eps=0.2, alpha=0.3, gamma=0.9):
        super().__init__(mark)
        self.archive = {}
        self.eps = eps; self.alpha = alpha; self.gamma = gamma
    def query_rune(self, state, act):
        if self.archive.get((state, act)) is None: self.archive[(state, act)] = 1.0
        return self.archive[(state, act)]
    def select_rune(self, state, acts):
        if random.random()<self.eps: return random.choice(acts)
        qs=[self.query_rune(state,a) for a in acts]
        m=max(qs)
        if qs.count(m)>1: idx=random.choice([i for i,q in enumerate(qs) if q==m])
        else: idx=qs.index(m)
        return acts[idx]
    def refine_rune(self, schema, acts, choice, over, augur):
        rew=0
        if over:
            win=augur.query_victor()
            rew = 20 if win==0 else 10 if win==self.mark else -20
        prev = schema.snapshot_previous()
        prev_q = self.query_rune(prev, choice)
        new_s = schema.snapshot_current()
        mx = max([self.query_rune(new_s,a) for a in acts])
        self.archive[(prev,choice)] = prev_q + self.alpha*((rew + self.gamma*mx) - prev_q)

class c4DiscToken:
    ORB_RADIUS = 30
    def __init__(self, mark):
        self.mark = mark
        self.surface = pygame.Surface((c4HexCell.RUNIC_CAPACITY-3, c4HexCell.RUNIC_CAPACITY-3))
        self.hue = BLUE if mark==1 else RED
    def anchor_coordinates(self, x,y): self.x0, self.y0 = x,y
    def define_column(self, c): self.col = c
    def reveal_column(self): return self.col
    def define_row(self, r): self.row = r
    def reveal_row(self): return self.row
    def shift_east(self, canvas, step=1):
        self.define_column(self.col+step)
        self.surface.fill(BLACK)
        canvas.blit(self.surface, (self.x0, self.y0))
        self.x0 += step * c4HexCell.RUNIC_CAPACITY
        self.depict_orb(canvas)
    def shift_west(self, canvas):
        self.define_column(self.col-1)
        self.surface.fill(BLACK)
        canvas.blit(self.surface,(self.x0,self.y0))
        self.x0 -= c4HexCell.RUNIC_CAPACITY
        self.depict_orb(canvas)
    def plummet_disc(self, canvas, r):
        self.define_row(r)
        self.surface.fill(BLACK)
        canvas.blit(self.surface,(self.x0,self.y0))
        self.y0 += (self.row+1)*c4HexCell.RUNIC_CAPACITY
        self.surface.fill(WHITE)
        canvas.blit(self.surface,(self.x0,self.y0))
        self.depict_orb(canvas)
    def reveal_type(self): return self.mark
    def depict_orb(self, canvas):
        pygame.draw.circle(self.surface, self.hue, (c4HexCell.RUNIC_CAPACITY//2, c4HexCell.RUNIC_CAPACITY//2), c4DiscToken.ORB_RADIUS)
        self.surface = self.surface.convert()
        canvas.blit(self.surface,(self.x0,self.y0))

class c4OracleMatrix:
    RUNE_SEQUENCE = 4
    def __init__(self, schema):
        self.schema = schema
        self.dim_r, self.dim_c = schema.schema_dimensions()
        self.victor = 0
    def assess_epoch(self):
        path, mark = self.schema.retrieve_recent()
        net = self.schema.glyph_network()
        hit = self.detect_sequence(path, net)
        if hit: self.victor = mark
        return hit or self.schema.verify_fullness()
    def detect_sequence(self, path, net):
        for r,c in path:
            nd = net[r][c]
            sc = [nd.top_left_score, nd.top_score, nd.top_right_score, nd.left_score,
                  nd.right_score, nd.bottom_left_score, nd.bottom_score, nd.bottom_right_score]
            if any(x==c4OracleMatrix.RUNE_SEQUENCE for x in sc): return True
        return False
    def pronounce_victor(self):
        return 'СИНІЙ' if self.victor==1 else 'ЧЕРВОНИЙ' if self.victor==2 else 'TIE'
    def query_victor(self): return self.victor

class c4NodeGlyph:
    def __init__(self):
        self.top_left = self.top = self.top_right = None
        self.left = self.right = None
        self.bottom_left = self.bottom = self.bottom_right = None
        self.top_left_score = self.top_score = self.top_right_score = 1
        self.left_score = self.right_score = 1
        self.bottom_left_score = self.bottom_score = self.bottom_right_score = 1
        self.value = 0
        self.visited = False
#
class ppAgentLumina:
    def __init__(self, x_rate=0.3, l_rate=0.2, d_factor=0.9):
        self.x_rate = x_rate
        self.l_rate = l_rate
        self.d_factor = d_factor
        self.table_store = {}
        self.prev_st = None
        self.prev_q = 0.0
        self.prev_act = None

    def reset_round(self):
        self.prev_st = None
        self.prev_q = 0.0
        self.prev_act = None

    def encode_state(self, state):
        a = int(state[0] / 50)
        b = int(state[1] / 30)
        c = 1 if state[2] > 0 else -1
        d = 1 if state[3] > 0 else -1
        e = int(state[4] / 30)
        return (a, b, c, d, e)

    def choose_move(self, state, moves):
        enc = self.encode_state(state)
        self.prev_st = enc
        if random.random() < self.x_rate:
            mv = random.choice(moves)
            self.prev_act = mv
            self.prev_q = self.fetch_q(enc, mv)
            return mv
        else:
            qs = [self.fetch_q(enc, mv) for mv in moves]
            mx = max(qs)
            if qs.count(mx) > 1:
                candidates = [i for i, q in enumerate(qs) if q == mx]
                idx = random.choice(candidates)
            else:
                idx = qs.index(mx)
            sel = moves[idx]
            self.prev_act = sel
            self.prev_q = self.fetch_q(enc, sel)
            return sel

    def fetch_q(self, state, action):
        if (state, action) not in self.table_store:
            self.table_store[(state, action)] = 0.0
        return self.table_store[(state, action)]

    def modify_q(self, reward, state, moves):
        enc = self.encode_state(state)
        qs = [self.fetch_q(enc, mv) for mv in moves]
        nxt = max(qs) if qs else 0.0
        self.table_store[(self.prev_st, self.prev_act)] = (
            self.prev_q + self.l_rate * ((reward + self.d_factor * nxt) - self.prev_q)
        )

    def export_table(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self.table_store, f, protocol=pickle.HIGHEST_PROTOCOL)

    def import_table(self, fname):
        with open(fname, 'rb') as f:
            self.table_store = pickle.load(f)

class ppHumanController:
    pass

class ppScrambleMover:
    def __init__(self):
        pass

    def decide_move(self, choices):
        return random.choice(choices)

class ppPaddleBattle:
    def __init__(self, w=800, h=600, training=False):
        self.w = w
        self.h = h
        self.pw = 15
        self.ph = 100
        self.bs = 15
        self.ps = 10
        self.bsx = 7
        self.bsy = 7
        self.training = training

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.a1_per_block = []
        self.a2_per_block = []
        self.itrack = []

        self.ball_pos = [w // 2, h // 2]
        self.ball_vel = [random.choice([-1, 1]) * self.bsx, random.choice([-1, 1]) * self.bsy]
        self.lp = [0, h // 2 - self.ph // 2]
        self.rp = [w - self.pw, h // 2 - self.ph // 2]

        self.over = False
        self.human = None
        self.comp = None
        self.hum_turn = None
        self.a1 = None
        self.a2 = None
        self.ai_flag = False
        self.trained = None

        self.a1_score = 0
        self.a2_score = 0
        self.win_size = 1000
        self.a1_rate = []
        self.a2_rate = []
        self.itrack = []

        self.h_score = 0
        self.ai_score = 0

        pygame.init()
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Пінг-Понг')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 30)
        self.bg = pygame.Surface(self.display.get_size()).convert()

    def main_interface(self, iters=5000):
        run = True
        while run:
            self.bg.fill((255, 255, 255))
            self.render_menu()
            pygame.display.flip()
            self.display.blit(self.bg, (0, 0))
            for evt in pygame.event.get():
                if evt.type == pygame.QUIT:
                    run = False
                elif evt.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    if self.r1.collidepoint(pos):
                        if self.trained is None:
                            self.trained = ppAgentLumina()
                            try:
                                self.trained.import_table("pong_agent2_qtable")
                                print("Q-table loaded successfully")
                            except FileNotFoundError:
                                print("Q-table not found. Please train the AI first.")
                                continue
                        self.trained.x_rate = 0
                        self.initiate_game(ppHumanController(), self.trained)
                        self.commence_game()
                    elif self.r2.collidepoint(pos):
                        self.a1 = ppAgentLumina()
                        self.a2 = ppAgentLumina()
                        self.initiate_training(self.a1, self.a2)
                        self.conduct_training(iters)
                        self.archive_tables()
                        self.trained = self.a2
                        self.training = False
                    elif self.r3.collidepoint(pos):
                        run = False
                elif evt.type == pygame.KEYDOWN and evt.key == pygame.K_ESCAPE:
                    run = False
            self.clock.tick(60)
        pygame.quit()

    def render_menu(self):
        fnt = pygame.font.SysFont('mono', 50, bold=True)
        tit = fnt.render('Пінг-Понг', True, (0, 0, 0))
        fw, fh = fnt.size('Пінг-Понг')
        self.bg.blit(tit, ((self.w - fw) // 2, 150))
        fn2 = pygame.font.SysFont('mono', 30, bold=True)
        t1, t2, t3 = 'Грати', 'Тренування', 'Вийти'
        ps1 = fn2.render(t1, True, (0, 0, 0))
        fw, fh = fn2.size(t1)
        self.r1 = ps1.get_rect(topleft=((self.w - fw) // 2, 300))
        self.bg.blit(ps1, ((self.w - fw) // 2, 300))
        ps2 = fn2.render(t2, True, (0, 0, 0))
        fw, fh = fn2.size(t2)
        self.r2 = ps2.get_rect(topleft=((self.w - fw) // 2, 350))
        self.bg.blit(ps2, ((self.w - fw) // 2, 350))
        ps3 = fn2.render(t3, True, (0, 0, 0))
        fw, fh = fn2.size(t3)
        self.r3 = ps3.get_rect(topleft=((self.w - fw) // 2, 400))
        self.bg.blit(ps3, ((self.w - fw) // 2, 400))

    def restart(self):
        self.ball_pos = [self.w // 2, self.h // 2]
        self.ball_vel = [random.choice([-1, 1]) * self.bsx, random.choice([-1, 1]) * self.bsy]
        self.lp = [0, self.h // 2 - self.ph // 2]
        self.rp = [self.w - self.pw, self.h // 2 - self.ph // 2]
        if not self.training:
            self.display.fill((0, 0, 0))
            self.refresh_display()

    def advance_state(self, left=None, right=None):
        if left == 'up' and self.lp[1] > 0:
            self.lp[1] -= self.ps
        elif left == 'down' and self.lp[1] < self.h - self.ph:
            self.lp[1] += self.ps
        if left == 'stay': pass
        if right == 'up' and self.rp[1] > 0:
            self.rp[1] -= self.ps
        elif right == 'down' and self.rp[1] < self.h - self.ph:
            self.rp[1] += self.ps
        if right == 'stay': pass
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        if self.ball_pos[1] <= 0 or self.ball_pos[1] >= self.h - self.bs:
            self.ball_vel[1] = -self.ball_vel[1]
        reward_l, reward_r, done = 0, 0, False
        if (self.ball_pos[0] <= self.pw and self.lp[1] <= self.ball_pos[1] <= self.lp[1] + self.ph):
            self.ball_vel[0] = -self.ball_vel[0]
            reward_l = 1
        if (self.ball_pos[0] >= self.w - self.pw - self.bs and self.rp[1] <= self.ball_pos[1] <= self.rp[1] + self.ph):
            self.ball_vel[0] = -self.ball_vel[0]
            reward_r = 1
        if self.ball_pos[0] < 0:
            self.a2_score += 1 if self.training else 0
            if not self.training:
                if self.human: self.ai_score += 1
                else: self.h_score += 1
            reward_r, reward_l, done = 5, -5, True
        elif self.ball_pos[0] > self.w:
            self.a1_score += 1 if self.training else 0
            if not self.training:
                if self.human: self.h_score += 1
                else: self.ai_score += 1
            reward_l, reward_r, done = 5, -5, True
        if self.ball_vel[0] < 0:
            diff = abs(self.ball_pos[1] - (self.lp[1] + self.ph // 2))
            prox = max(0, 0.2 - 0.001 * diff)
            reward_l += prox
        else:
            diff = abs(self.ball_pos[1] - (self.rp[1] + self.ph // 2))
            prox = max(0, 0.2 - 0.001 * diff)
            reward_r += prox
        return reward_l, reward_r, done

    def draw_frame(self):
        if not self.training:
            self.display.fill((0, 0, 0))
            pygame.draw.rect(self.display, (255, 255, 255), (self.lp[0], self.lp[1], self.pw, self.ph))
            pygame.draw.rect(self.display, (255, 255, 255), (self.rp[0], self.rp[1], self.pw, self.ph))
            pygame.draw.rect(self.display, (255, 255, 255), (self.ball_pos[0], self.ball_pos[1], self.bs, self.bs))
            for i in range(0, self.h, 20):
                pygame.draw.rect(self.display, (255, 255, 255), (self.w // 2 - 1, i, 2, 10))
            ls = self.font.render(str(self.h_score if self.human else self.ai_score), True, (255, 255, 255))
            rs = self.font.render(str(self.ai_score if self.human else self.h_score), True, (255, 255, 255))
            self.display.blit(ls, (self.w // 4, 20))
            self.display.blit(rs, (3 * self.w // 4, 20))

    def refresh_display(self):
        if not self.training:
            self.draw_frame()
            pygame.display.flip()
            self.clock.tick(60)

    def available_moves(self):
        return ['up', 'down', 'stay']

    def current_state(self, left_flag):
        if left_flag:
            return [self.ball_pos[0], self.ball_pos[1], self.ball_vel[0], self.ball_vel[1], self.lp[1]]
        else:
            return [self.w - self.ball_pos[0], self.ball_pos[1], -self.ball_vel[0], self.ball_vel[1], self.rp[1]]

    def initiate_training(self, a1, a2):
        if isinstance(a1, ppAgentLumina) and isinstance(a2, ppAgentLumina):
            self.training = True
            self.a1 = a1
            self.a2 = a2

    def track_training(self):
        display.clear_output(wait=True)
        self.ax.clear()
        self.ax.plot(self.itrack, self.a1_per_block, label='Left Paddle Wins')
        self.ax.plot(self.itrack, self.a2_per_block, label='Right Paddle Wins')
        self.ax.set_title('Training Progress per Block')
        self.ax.set_xlabel('Training Episodes')
        self.ax.set_ylabel('Wins per Block')
        self.ax.legend()
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax.set_ylim(bottom=0)
        self.ax.grid(True)
        self.fig.canvas.draw()
        plt.pause(0.1)

    def conduct_training(self, iters):
        if not self.training:
            return
        self.a1_score = 0
        self.a2_score = 0
        self.a1_per_block.clear()
        self.a2_per_block.clear()
        self.itrack.clear()
        w1, w2 = 0, 0
        for i in range(iters):
            if i % 1000 == 0 and i > 0:
                print(f"Training iteration {i}/{iters}")
            self.a1.reset_round()
            self.a2.reset_round()
            self.restart()
            done = False
            steps = 0
            max_s = 2000
            while not done and steps < max_s:
                la = self.a1.choose_move(self.current_state(True), self.available_moves())
                ra = self.a2.choose_move(self.current_state(False), self.available_moves())
                rl, rr, done = self.advance_state(la, ra)
                self.a1.modify_q(rl, self.current_state(True), self.available_moves())
                self.a2.modify_q(rr, self.current_state(False), self.available_moves())
                steps += 1
            if rl > rr:
                w1 += 1
                self.a1_score += 1
            else:
                w2 += 1
                self.a2_score += 1
            if (i + 1) % self.win_size == 0:
                self.a1_per_block.append(w1)
                self.a2_per_block.append(w2)
                self.itrack.append(i + 1)
                self.track_training()
                w1, w2 = 0, 0
        if w1 + w2 > 0:
            self.a1_per_block.append(w1)
            self.a2_per_block.append(w2)
            self.itrack.append(iters)
            self.track_training()
        print("Training complete!")
        print(f"Total wins — Left: {self.a1_score}, Right: {self.a2_score}")
        self.fig.savefig('pong_training_progress.png')

    def archive_tables(self):
        self.a1.export_table("pong_agent1_qtable")
        self.a2.export_table("pong_agent2_qtable")

    def initiate_game(self, p_left, p_right):
        if isinstance(p_left, ppHumanController):
            self.human = True
            self.comp = False
            if isinstance(p_right, ppAgentLumina):
                self.ai = p_right
                self.ai.import_table("pong_agent2_qtable")
                self.ai.x_rate = 0
                self.ai_flag = True
            elif isinstance(p_right, ppScrambleMover):
                self.ai = p_right
                self.ai_flag = False
        elif isinstance(p_right, ppHumanController):
            self.human = False
            self.comp = True
            if isinstance(p_left, ppAgentLumina):
                self.ai = p_left
                self.ai.import_table("pong_agent1_qtable")
                self.ai.x_rate = 0
                self.ai_flag = True
            elif isinstance(p_left, ppScrambleMover):
                self.ai = p_left
                self.ai_flag = False

    def commence_game(self):
        if self.training:
            return
        run = True
        pygame.event.clear()
        self.restart()
        while run:
            for evt in pygame.event.get():
                if evt.type == pygame.QUIT:
                    run = False
            keys = pygame.key.get_pressed()
            act = 'stay'
            if self.human:
                if keys[pygame.K_w] and self.lp[1] > 0:
                    act = 'up'
                elif keys[pygame.K_s] and self.lp[1] < self.h - self.ph:
                    act = 'down'
                ai_act = (self.ai.choose_move(self.current_state(False), self.available_moves())
                          if self.ai_flag else self.ai.decide_move(self.available_moves()))
                rl, rr, done = self.advance_state(act, ai_act)
            else:
                if keys[pygame.K_UP] and self.rp[1] > 0:
                    act = 'up'
                elif keys[pygame.K_DOWN] and self.rp[1] < self.h - self.ph:
                    act = 'down'
                ai_act = (self.ai.choose_move(self.current_state(True), self.available_moves())
                          if self.ai_flag else self.ai.decide_move(self.available_moves()))
                rl, rr, done = self.advance_state(ai_act, act)
            self.refresh_display()
            self.clock.tick(60)
            if done:
                time.sleep(1)
                self.restart()

# 

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 100
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.1
GAMMA = 0.9

class snkDirection(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

snkPoint = namedtuple('Point', 'x, y')

class snkSnakeGameAI:
    def __init__(self, m_w=800, m_h=600, g_w=640, g_h=480):
        self.m_w = m_w
        self.m_h = m_h
        self.g_w = g_w
        self.g_h = g_h
        self.c_w = self.m_w
        self.c_h = self.m_h
        self.disp = pygame.display.set_mode((self.c_w, self.c_h))
        pygame.display.set_caption('Змійка')
        self.clk = pygame.time.Clock()
        self.bg = pygame.Surface(self.disp.get_size()).convert()
        self.reinit_state()
        pygame.init()
        font = pygame.font.Font(None, 25)  

    def reinit_state(self):
        self.dirn = snkDirection.RIGHT
        self.hd = snkPoint(self.g_w/2, self.g_h/2)
        self.trail = [self.hd,
                      snkPoint(self.hd.x-BLOCK_SIZE, self.hd.y),
                      snkPoint(self.hd.x-(2*BLOCK_SIZE), self.hd.y)]
        self.scr = 0
        self.food_pt = None
        self._snk_place()
        self.iter_count = 0

    def _snk_place(self):
        px = random.randint(0, (self.g_w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        py = random.randint(0, (self.g_h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food_pt = snkPoint(px, py)
        if self.food_pt in self.trail:
            self._snk_place()

    def process_step(self, act_cmd):
        self.iter_count += 1
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                quit()
        self._advance(act_cmd)
        self.trail.insert(0, self.hd)
        rew = 0
        over = False
        if self.check_collision() or self.iter_count > 100 * len(self.trail):
            over = True
            rew = -10
            return rew, over, self.scr
        if self.hd == self.food_pt:
            self.scr += 1
            rew = 10
            self._snk_place()
        else:
            self.trail.pop()
        self.refresh_ui()
        self.clk.tick(SPEED)
        return rew, over, self.scr

    def check_collision(self, pt=None):
        if pt is None:
            pt = self.hd
        if pt.x > self.g_w - BLOCK_SIZE or pt.x < 0 or pt.y > self.g_h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.trail[1:]:
            return True
        return False

    def refresh_ui(self):
        pygame.init()
        font = pygame.font.Font(None, 25)  
        self.disp.fill(BLACK)
        for p in self.trail:
            pygame.draw.rect(self.disp, BLUE1, pygame.Rect(p.x, p.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.disp, BLUE2, pygame.Rect(p.x+4, p.y+4, 12, 12))
        pygame.draw.rect(self.disp, RED, pygame.Rect(self.food_pt.x, self.food_pt.y, BLOCK_SIZE, BLOCK_SIZE))
        txt = font.render("Рахунок: " + str(self.scr), True, WHITE)
        self.disp.blit(txt, [0, 0])
        pygame.display.flip()

    def _advance(self, act_cmd):
        seq = [snkDirection.RIGHT, snkDirection.DOWN, snkDirection.LEFT, snkDirection.UP]
        idx = seq.index(self.dirn)
        if np.array_equal(act_cmd, [1, 0, 0]):
            ndir = seq[idx]
        elif np.array_equal(act_cmd, [0, 1, 0]):
            ndir = seq[(idx + 1) % 4]
        else:
            ndir = seq[(idx - 1) % 4]

        self.dirn = ndir
        x, y = self.hd.x, self.hd.y
        if ndir == snkDirection.RIGHT:
            x += BLOCK_SIZE
        elif ndir == snkDirection.LEFT:
            x -= BLOCK_SIZE
        elif ndir == snkDirection.DOWN:
            y += BLOCK_SIZE
        elif ndir == snkDirection.UP:
            y -= BLOCK_SIZE

        self.hd = snkPoint(x, y)

    def menu_main(self):
        running = True
        self.c_w = self.m_w
        self.c_h = self.m_h
        self.disp = pygame.display.set_mode((self.c_w, self.c_h))
        self.bg = pygame.Surface(self.disp.get_size()).convert()
        self.bg.fill(WHITE)
        self.render_menu()
        pygame.display.flip()
        self.disp.blit(self.bg, (0, 0))
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    if self.rectA.collidepoint(pos):
                        self.c_w = self.g_w
                        self.c_h = self.g_h
                        self.disp = pygame.display.set_mode((self.c_w, self.c_h))
                        self.learn()
                        running = False
                    elif self.rectB.collidepoint(pos):
                        running = False
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    running = False
            pygame.display.flip()
            self.disp.blit(self.bg, (0, 0))
            self.clk.tick(60)
        pygame.quit()

    def render_menu(self):
        pygame.init()
        font = pygame.font.Font(None, 25)  
        fn = pygame.font.SysFont('mono', 50, bold=True)
        title = fn.render('ЗМІЙКА', True, BLACK)
        w_t, h_t = fn.size('ЗМІЙКА')
        self.bg.blit(title, ((self.c_w - w_t) // 2, 150))
        fn2 = pygame.font.SysFont('mono', 30, bold=True)
        txtA = 'Грати'
        txtB = 'Вийти'
        surfA = fn2.render(txtA, True, BLACK)
        wA, hA = fn2.size(txtA)
        self.rectA = surfA.get_rect(topleft=((self.c_w - wA) // 2, 300))
        self.bg.blit(surfA, ((self.c_w - wA) // 2, 300))
        surfB = fn2.render(txtB, True, BLACK)
        wB, hB = fn2.size(txtB)
        self.rectB = surfB.get_rect(topleft=((self.c_w - wB) // 2, 350))
        self.bg.blit(surfB, ((self.c_w - wB) // 2, 350))

    def learn(self):
        plt.ion()
        s_list, m_list = [], []
        tot = 0
        rec = 0
        ag = snkAgent()
        while True:
            st_old = ag.get_state(self)
            mv = ag.get_action(st_old)
            rw, end, sc = self.process_step(mv)
            st_new = ag.get_state(self)
            ag.train_short_memory(st_old, mv, rw, st_new, end)
            ag.remember(st_old, mv, rw, st_new, end)
            if end:
                self.reinit_state()
                ag.game_count += 1
                ag.train_long_memory()
                if sc > rec:
                    rec = sc
                print('Game', ag.game_count, 'Score', sc, 'Record:', rec)
                s_list.append(sc)
                tot += sc
                m_list.append(tot / ag.game_count)
                self.graph(s_list, m_list)

    def graph(self, scs, mscs):
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(scs)
        plt.plot(mscs)
        plt.ylim(ymin=0)
        plt.text(len(scs)-1, scs[-1], str(scs[-1]))
        plt.text(len(mscs)-1, mscs[-1], str(mscs[-1]))
        plt.show(block=False)
        plt.pause(.1)

class snkAgent:
    def __init__(self):
        self.game_count = 0
        self.eps = 0
        self.gam = GAMMA
        self.mem = deque(maxlen=MAX_MEMORY)
        self.qt = {}
        self.learn_rate = LR

    def get_state(self, gm):
        hd = gm.trail[0]
        p_l = snkPoint(hd.x - BLOCK_SIZE, hd.y)
        p_r = snkPoint(hd.x + BLOCK_SIZE, hd.y)
        p_u = snkPoint(hd.x, hd.y - BLOCK_SIZE)
        p_d = snkPoint(hd.x, hd.y + BLOCK_SIZE)
        d_l = gm.dirn == snkDirection.LEFT
        d_r = gm.dirn == snkDirection.RIGHT
        d_u = gm.dirn == snkDirection.UP
        d_d = gm.dirn == snkDirection.DOWN
        state = [
            (d_r and gm.check_collision(p_r)) or (d_l and gm.check_collision(p_l)) or
            (d_u and gm.check_collision(p_u)) or (d_d and gm.check_collision(p_d)),
            (d_u and gm.check_collision(p_r)) or (d_d and gm.check_collision(p_l)) or
            (d_l and gm.check_collision(p_u)) or (d_r and gm.check_collision(p_d)),
            (d_d and gm.check_collision(p_r)) or (d_u and gm.check_collision(p_l)) or
            (d_r and gm.check_collision(p_u)) or (d_l and gm.check_collision(p_d)),
            d_l, d_r, d_u, d_d,
            gm.food_pt.x < hd.x,
            gm.food_pt.x > hd.x,
            gm.food_pt.y < hd.y,
            gm.food_pt.y > hd.y
        ]
        return tuple(state)

    def remember(self, st, mv, rw, nxt, end):
        self.mem.append((st, mv, rw, nxt, end))

    def train_short_memory(self, st, mv, rw, nxt, end):
        if st not in self.qt:
            self.qt[st] = [0.0, 0.0, 0.0]
        if nxt not in self.qt:
            self.qt[nxt] = [0.0, 0.0, 0.0]
        idx = np.argmax(mv)
        old = self.qt[st][idx]
        nxt_max = max(self.qt[nxt]) if not end else 0
        new = old + self.learn_rate * (rw + self.gam * nxt_max - old)
        self.qt[st][idx] = new

    def train_long_memory(self):
        sample = random.sample(self.mem, BATCH_SIZE) if len(self.mem) > BATCH_SIZE else self.mem
        for st, mv, rw, nxt, end in sample:
            self.train_short_memory(st, mv, rw, nxt, end)

    def get_action(self, st):
        self.eps = 80 - self.game_count
        move_cmd = [0, 0, 0]
        if st not in self.qt:
            self.qt[st] = [0.0, 0.0, 0.0]
        if random.randint(0, 200) < self.eps:
            i = random.randint(0, 2)
            move_cmd[i] = 1
        else:
            i = np.argmax(self.qt[st])
            move_cmd[i] = 1
        return move_cmd

# 

class tttLearningAgent:
    def __init__(self, exp_rate_val=0.2, lr_val=0.3, df_val=0.9):
        self.exp_rate_val = exp_rate_val
        self.lr_val = lr_val
        self.df_val = df_val
        self.q_tab = {} 
        self.prev_state = None
        self.prev_q = 0.0
        self.prev_state_act = None

    def reset_session(self):
        self.prev_state = None
        self.prev_q = 0.0
        self.prev_state_act = None

    def choose_move(self, state, available_actions): 
        self.prev_state = tuple(state)
        if random.random() < self.exp_rate_val:
            choice = random.choice(available_actions)
            self.prev_state_act = (self.prev_state, choice)
            self.prev_q = self.retrieve_q(self.prev_state, choice)
            return choice
        else:
            q_list = [self.retrieve_q(self.prev_state, act) for act in available_actions]
            best_q = max(q_list)
            if q_list.count(best_q) > 1:
                options = [i for i, v in enumerate(q_list) if v == best_q]
                idx = random.choice(options)
            else:
                idx = q_list.index(best_q)
            sel = available_actions[idx]
            self.prev_state_act = (self.prev_state, sel)
            self.prev_q = self.retrieve_q(self.prev_state, sel)
            return sel

    def retrieve_q(self, state, action):
        if self.q_tab.get((state, action)) is None:
            self.q_tab[(state, action)] = 1.0
        return self.q_tab[(state, action)]

    def adjust_q(self, reward, state, available_actions):
        next_qs = [self.retrieve_q(tuple(state), act) for act in available_actions]
        max_nq = max(next_qs) if next_qs else 0.0
        self.q_tab[self.prev_state_act] = self.prev_q + self.lr_val * ((reward + self.df_val * max_nq) - self.prev_q)

    def export_q_table(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.q_tab, f, protocol=pickle.HIGHEST_PROTOCOL)

    def import_q_table(self, file_name):
        with open(file_name, 'rb') as f:
            self.q_tab = pickle.load(f)

class tttHumanPlayer:
    pass

class tttRandomPlayer:
    def __init__(self):
        pass
    
    def make_random_move(self, possible_moves):
        return random.choice(possible_moves)

class tttTicTacToeGame:
    def __init__(self, training_mode=False):
        self.board_state = [' '] * 9
        self.game_done = False
        self.human_flag = None
        self.ai_flag = None
        self.human_turn_flag = None
        self.training_mode = training_mode
        self.agent_a = None
        self.agent_b = None
        self.ai_player = None
        self.is_ai_mode = False
        
        self.agent_a_wins = 0
        self.agent_b_wins = 0
        self.draw_count = 0
        self.window_size = 1000
        self.history_a = []
        self.history_b = []
        self.history_draw = []
        self.iter_tracked = []
        plt.ion()
        self.figure, self.axes = plt.subplots(figsize=(10,6))
        self.block_a = []
        self.block_b = []
        self.block_draw = []
        self.iter_tracked_block = []
        
        self.score_human = 0
        self.score_ai = 0
        self.score_draw = 0
        
        pygame.init()
        self.display_surf = pygame.display.set_mode((800, 600))
        pygame.display.set_caption('Хрестики-Нулики')
        self.bg = pygame.Surface(self.display_surf.get_size()).convert()

    def display_menu(self, iterations=200000):
        self.iterations = iterations
        menu_active = True
        start_flag = False
        self.bg.fill((255, 255, 255))
        self.render_menu()
        
        while menu_active:
            for evt in pygame.event.get():
                if evt.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    if self.btn1.collidepoint(pos):
                        start_flag = True
                        menu_active = False
                        mode = "single"
                    elif self.btn2.collidepoint(pos):
                        start_flag = True
                        menu_active = False
                        mode = "train"
                    elif self.btn3.collidepoint(pos):
                        menu_active = False
                if evt.type == pygame.KEYDOWN and evt.key == pygame.K_ESCAPE:
                    menu_active = False
            pygame.display.flip()
            self.display_surf.blit(self.bg, (0, 0))
        
        if not start_flag:
            pygame.quit()
        elif mode == "train":
            self.training_mode = True
            self.configure_training(tttLearningAgent(), tttLearningAgent())
            self.run_training(self.iterations)
            self.store_q_tables()
            self.training_mode = False
            self.display_menu()
        else:
            self.training_mode = False
            self.configure_game(tttHumanPlayer(), tttLearningAgent())
            self.initialize_board()
            self.start_game()

    def render_menu(self):
        font1 = pygame.font.SysFont('mono', 50, bold=True)
        title = font1.render('Хрестики-Нулики', True, (0, 0, 0))
        w, h = font1.size('Хрестики-Нулики')
        self.bg.blit(title, ((800 - w)//2, 150))
        
        font2 = pygame.font.SysFont('mono', 30, bold=True)
        txt1 = 'Грати'
        txt2 = 'Тренування'
        txt3 = 'Вийти'
        
        surf1 = font2.render(txt1, True, (0, 0, 0))
        w1, h1 = font2.size(txt1)
        self.btn1 = surf1.get_rect(topleft=((800 - w1)//2, 300))
        self.bg.blit(surf1, ((800 - w1)//2, 300))
        
        surf2 = font2.render(txt2, True, (0, 0, 0))
        w2, h2 = font2.size(txt2)
        self.btn2 = surf2.get_rect(topleft=((800 - w2)//2, 350))
        self.bg.blit(surf2, ((800 - w2)//2, 350))
        
        surf3 = font2.render(txt3, True, (0, 0, 0))
        w3, h3 = font2.size(txt3)
        self.btn3 = surf3.get_rect(topleft=((800 - w3)//2, 400))
        self.bg.blit(surf3, ((800 - w3)//2, 400))

    def initialize_board(self):
        if self.training_mode:
            self.board_state = [' '] * 9
            return
        self.board_state = [' '] * 9
        self.human_turn_flag = random.choice([True, False])
        self.surface = pygame.Surface(self.display_surf.get_size()).convert()
        self.surface.fill((250, 250, 250))
        grid = 450
        cell = grid // 3
        ox = (800 - grid)//2
        oy = (600 - grid - 75)//2
        pygame.draw.line(self.surface, (0,0,0), (ox, oy+cell), (ox+grid, oy+cell), 2)
        pygame.draw.line(self.surface, (0,0,0), (ox, oy+2*cell), (ox+grid, oy+2*cell), 2)
        pygame.draw.line(self.surface, (0,0,0), (ox+cell, oy), (ox+cell, oy+grid), 2)
        pygame.draw.line(self.surface, (0,0,0), (ox+2*cell, oy), (ox+2*cell, oy+grid), 2)
        self.refresh_scoreboard()

    def refresh_scoreboard(self):
        self.surface.fill((250,250,250), (0, 525, 800, 75))
        pygame.draw.line(self.surface, (0,0,0), (0,525), (800,525), 2)
        fnt = pygame.font.Font(None, 30)
        txt_h = fnt.render(f"Людина: {self.score_human}", 1, (10,10,10))
        txt_a = fnt.render(f"ШІ: {self.score_ai}", 1, (10,10,10))
        txt_d = fnt.render(f"Нічия: {self.score_draw}", 1, (10,10,10))
        self.surface.blit(txt_h, (50,535))
        self.surface.blit(txt_a, (350,535))
        self.surface.blit(txt_d, (650,535))

    def assess_win(self, symbol):
        for i in range(3):
            if symbol == self.board_state[i*3] == self.board_state[i*3+1] == self.board_state[i*3+2]:
                return 1.0, True
        for i in range(3):
            if symbol == self.board_state[i] == self.board_state[i+3] == self.board_state[i+6]:
                return 1.0, True
        if symbol == self.board_state[0] == self.board_state[4] == self.board_state[8]:
            return 1.0, True
        if symbol == self.board_state[2] == self.board_state[4] == self.board_state[6]:
            return 1.0, True
        if not any(c == ' ' for c in self.board_state):
            return 0.5, True
        return 0.0, False

    def list_moves(self):
        return [i+1 for i, v in enumerate(self.board_state) if v == ' ']

    def apply_move(self, is_x, pos):
        mark = 'X' if is_x else '0'
        if self.board_state[pos-1] != ' ':
            return -5, True
        self.board_state[pos-1] = mark
        return self.assess_win(mark)

    def render_move(self, pos, is_x):
        grid = 450
        cell = grid // 3
        ox = (800 - grid)//2
        oy = (600 - grid - 75)//2
        row = (pos-1)//3
        col = (pos-1)%3
        cx = ox + col*cell + cell//2 -12
        cy = oy + row*cell + cell//2 -12
        reward, done = self.apply_move(is_x, pos)
        if reward == -5:
            fnt = pygame.font.Font(None,30)
            t = fnt.render('Заборонений хід!',1,(10,10,10))
            self.surface.fill((250,250,250),(0,525,800,30))
            self.surface.blit(t,(50,535))
            return reward, done
        fnt2 = pygame.font.Font(None,60)
        if is_x:
            txt = fnt2.render('X',1,(10,10,10))
            self.surface.blit(txt,(cx,cy))
            if self.human_flag and reward==1:
                txt2 = fnt2.render('Людина виграла!',1,(10,10,10))
                self.surface.fill((250,250,250),(0,525,800,30))
                self.surface.blit(txt2,(50,535))
                self.score_human +=1
                self.refresh_scoreboard()
            elif self.ai_flag and reward==1:
                txt2 = fnt2.render("Комп'ютер виграв!",1,(10,10,10))
                self.surface.fill((250,250,250),(0,525,800,30))
                self.surface.blit(txt2,(50,535))
                self.score_ai +=1
                self.refresh_scoreboard()
        else:
            txt = fnt2.render('O',1,(10,10,10))
            self.surface.blit(txt,(cx,cy))
            if not self.human_flag and reward==1:
                txt2 = fnt2.render('Людина виграла!',1,(10,10,10))
                self.surface.fill((250,250,250),(0,525,800,30))
                self.surface.blit(txt2,(50,535))
                self.score_human +=1
                self.refresh_scoreboard()
            elif not self.ai_flag and reward==1:
                txt2 = fnt2.render("Комп'ютер виграв!",1,(10,10,10))
                self.surface.fill((250,250,250),(0,525,800,30))
                self.surface.blit(txt2,(50,535))
                self.score_ai +=1
                self.refresh_scoreboard()
        if reward == 0.5:
            fnt3 = pygame.font.Font(None,30)
            td = fnt3.render('Нічия!',1,(10,10,10))
            self.surface.fill((250,250,250),(0,525,800,30))
            self.surface.blit(td,(50,535))
            self.score_draw +=1
            self.refresh_scoreboard()
        return reward, done

    def fetch_mouse_pos(self):
        mx, my = pygame.mouse.get_pos()
        grid = 450
        cell = grid//3
        ox = (800-grid)//2
        oy = (600-grid-75)//2
        if my>525 or mx<ox or mx>ox+grid or my<oy:
            return -1
        r = (my-oy)//cell
        c = (mx-ox)//cell
        return r*3 + c + 1

    def process_move(self, is_x):
        pos = self.fetch_mouse_pos()
        if pos == -1:
            return 0, False
        return self.render_move(pos, is_x)

    def refresh_display(self):
        self.display_surf.blit(self.surface, (0,0))
        pygame.display.flip()

    def configure_training(self, ag_a, ag_b):
        if isinstance(ag_a, tttLearningAgent) and isinstance(ag_b, tttLearningAgent):
            self.training_mode = True
            self.agent_a, self.agent_b = ag_a, ag_b

    def plot_progress(self):
        display.clear_output(wait=True)
        self.axes.clear()
        self.axes.plot(self.iter_tracked_block, self.block_a, label='Agent A Wins')
        self.axes.plot(self.iter_tracked_block, self.block_b, label='Agent B Wins')
        self.axes.plot(self.iter_tracked_block, self.block_draw, label='Draws')
        self.axes.set_title('Training Progress')
        self.axes.set_xlabel('Iterations')
        self.axes.set_ylabel('Games')
        self.axes.legend()
        self.axes.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.axes.set_ylim(bottom=0)
        self.axes.grid(True)
        self.figure.canvas.draw()
        plt.pause(0.1)

    def run_training(self, iterations):
        if not self.training_mode:
            return
        self.agent_a_wins = self.agent_b_wins = self.draw_count = 0
        self.block_a.clear(); self.block_b.clear(); self.block_draw.clear(); self.iter_tracked_block.clear()
        win_a = win_b = draw_w = 0
        for i in range(iterations):
            if i%1000==0 and i>0:
                print(f"Training iteration {i}/{iterations}")
            self.agent_a.reset_session(); self.agent_b.reset_session(); self.initialize_board()
            done = False
            turn_x = random.choice([True, False])
            while not done:
                if turn_x:
                    mv = self.agent_a.choose_move(self.board_state, self.list_moves())
                else:
                    mv = self.agent_b.choose_move(self.board_state, self.list_moves())
                reward, done = self.apply_move(turn_x, mv)
                if reward==1.0:
                    if turn_x:
                        self.agent_a.adjust_q(reward, self.board_state, self.list_moves())
                        self.agent_b.adjust_q(-reward, self.board_state, self.list_moves())
                        self.agent_a_wins +=1; win_a+=1
                    else:
                        self.agent_a.adjust_q(-reward, self.board_state, self.list_moves())
                        self.agent_b.adjust_q(reward, self.board_state, self.list_moves())
                        self.agent_b_wins +=1; win_b+=1
                elif reward==0.5:
                    self.agent_a.adjust_q(reward, self.board_state, self.list_moves())
                    self.agent_b.adjust_q(reward, self.board_state, self.list_moves())
                    self.draw_count+=1; draw_w+=1
                elif reward==-5:
                    if turn_x: self.agent_a.adjust_q(reward, self.board_state, self.list_moves())
                    else: self.agent_b.adjust_q(reward, self.board_state, self.list_moves())
                else:
                    if turn_x: self.agent_b.adjust_q(reward, self.board_state, self.list_moves())
                    else: self.agent_a.adjust_q(reward, self.board_state, self.list_moves())
                turn_x = not turn_x
            if (i+1)%self.window_size==0:
                self.block_a.append(win_a); self.block_b.append(win_b); self.block_draw.append(draw_w);
                self.iter_tracked_block.append(i+1)
                self.plot_progress()
                win_a=win_b=draw_w=0
        print("Training complete!")
        print(f"Total: A={self.agent_a_wins}, B={self.agent_b_wins}, Draws={self.draw_count}")
        if win_a+win_b+draw_w>0:
            self.block_a.append(win_a); self.block_b.append(win_b); self.block_draw.append(draw_w);
            self.iter_tracked_block.append(iterations)
            self.plot_progress()
            plt.savefig('training_progress_per_block.png')

    def store_q_tables(self):
        self.agent_a.export_q_table("agentA_qtable")
        self.agent_b.export_q_table("agentB_qtable")

    def configure_game(self, p_x, p_o):
        if isinstance(p_x, tttHumanPlayer):
            self.human_flag, self.ai_flag = True, False
            if isinstance(p_o, tttLearningAgent):
                self.ai_player = p_o
                self.ai_player.import_q_table("agentB_qtable")
                self.ai_player.exp_rate_val = 0
                self.is_ai_mode = True
            elif isinstance(p_o, tttRandomPlayer):
                self.ai_player = p_o; self.is_ai_mode = False
        elif isinstance(p_o, tttHumanPlayer):
            self.human_flag, self.ai_flag = False, True
            if isinstance(p_x, tttLearningAgent):
                self.ai_player = p_x
                self.ai_player.import_q_table("agentA_qtable")
                self.ai_player.exp_rate_val = 0
                self.is_ai_mode = True
            elif isinstance(p_x, tttRandomPlayer):
                self.ai_player = p_x; self.is_ai_mode = False

    def start_game(self):
        running = True
        while running:
            if self.human_turn_flag:
                evt = pygame.event.wait()
                while evt.type != pygame.MOUSEBUTTONDOWN:
                    evt = pygame.event.wait()
                    self.refresh_display()
                    if evt.type == pygame.QUIT:
                        running = False; break
                rw, done = self.process_move(self.human_flag)
                self.refresh_display()
                if done:
                    time.sleep(1); self.initialize_board()
            else:
                if self.is_ai_mode:
                    mv = self.ai_player.choose_move(self.board_state, self.list_moves())
                    rw, done = self.render_move(mv, self.ai_flag); self.refresh_display()
                else:
                    mv = self.ai_player.make_random_move(self.list_moves())
                    rw, done = self.render_move(mv, self.ai_flag); self.refresh_display()
                if done:
                    time.sleep(1); self.initialize_board()
            self.human_turn_flag = not self.human_turn_flag

def execute_training():
    gm = tttTicTacToeGame(True)
    a1 = tttLearningAgent()
    a2 = tttLearningAgent()
    gm.configure_training(a1, a2)
    gm.run_training(200000)
    gm.store_q_tables()


def launch_game():
    gm = tttTicTacToeGame()
    hu = tttHumanPlayer()
    ai = tttLearningAgent()
    gm.configure_game(hu, ai)
    gm.initialize_board()
    gm.start_game()

#

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Меню гри")
background = pygame.Surface(screen.get_size()).convert()

def render_text(text, size, bold=False):
    pygame.init()
    font = pygame.font.Font(None, 25)  
    font = pygame.font.SysFont('mono', size, bold=bold)
    return font.render(text, True, BLACK)

def load_and_scale_image(path, size):
    try:
        image = pygame.image.load(path).convert_alpha()
        return pygame.transform.scale(image, (size - 20, size - 20))
    except FileNotFoundError:
        print(f"Image not found: {path}")
        return pygame.Surface((size - 20, size - 20))  

def run_pong():  
    game = ppPaddleBattle()
    game.main_interface()

def run_connect4():
    parser = argparse.ArgumentParser()
    parser.add_argument('iterations', nargs='?', default=1000, help="Number of training iterations")
    args = parser.parse_args()
    game = c4VistaEngine(900, 600)
    game.helm_overview(int(args.iterations))

def run_snake():
    game = snkSnakeGameAI()
    game.menu_main()

def run_tictactoe():
    game = tttTicTacToeGame()
    game.display_menu()

def draw_menu():
    background.fill(WHITE)
    
    title_surface = render_text('МЕНЮ', 50, bold=True)
    fw, fh = title_surface.get_size()
    background.blit(title_surface, ((WIDTH - fw) // 2, 50))
    
    start_x = (WIDTH - (2 * BUTTON_SIZE + BUTTON_SPACING)) // 2
    start_y = (HEIGHT - (2 * BUTTON_SIZE + BUTTON_SPACING)) // 2 + 50
    games = ["Пінг-Понг", "З'єднай 4", "Змійка", "Хрестики-Нулики"]
    button_rects = []
    image_paths = [
        "./pong_icon.png",      
        "./connect4_icon.png",  
        "./snake_icon.png",     
        "./tictactoe_icon.png" 
    ]
    
    images = [load_and_scale_image(path, BUTTON_SIZE) for path in image_paths]
    
    for i, (game, image) in enumerate(zip(games, images)):
        row = i // 2
        col = i % 2
        button_x = start_x + col * (BUTTON_SIZE + BUTTON_SPACING)
        button_y = start_y + row * (BUTTON_SIZE + BUTTON_SPACING)
        
        button_rect = pygame.Rect(button_x, button_y, BUTTON_SIZE, BUTTON_SIZE)
        pygame.draw.rect(background, BLACK, button_rect, 2, BUTTON_RADIUS)
        button_rects.append(button_rect)
        
        img_x = button_x + (BUTTON_SIZE - image.get_width()) // 2
        img_y = button_y + (BUTTON_SIZE - image.get_height()) // 2
        background.blit(image, (img_x, img_y))
        
        label_text = render_text(game, 20, bold=True)
        lw, lh = label_text.get_size()
        background.blit(label_text, (button_x + (BUTTON_SIZE - lw) // 2, 
                                   button_y + BUTTON_SIZE + 10))
    
    return button_rects

def main_menu():
    global screen, background  
    running = True
    clock = pygame.time.Clock()
    button_rects = draw_menu()
    game_functions = [run_pong, run_connect4, run_snake, run_tictactoe]
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                for i, rect in enumerate(button_rects):
                    if rect.collidepoint(pos):
                        print(f"Launching {['Pong', 'Connect 4', 'Snake', 'Tic-Tac-Toe'][i]}")
                        pygame.quit()  
                        game_functions[i]()  
                        pygame.init() 
                        screen = pygame.display.set_mode((WIDTH, HEIGHT)) 
                        pygame.display.set_caption("Меню гри")
                        background = pygame.Surface(screen.get_size()).convert() 
                        button_rects = draw_menu()  
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        
        screen.blit(background, (0, 0))
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main_menu()