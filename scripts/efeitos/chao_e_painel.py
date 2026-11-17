"""CHÃO + PAINEL: texto branco deitado no CHÃO (perspectiva) e palavras GIGANTES num painel ATRÁS da pessoa
(ela fica na frente), com um pico de muito TEXTO. Câmera parada (tripé).
uso (na pasta da cena): python3 <skill>/scripts/efeitos/chao_e_painel.py [--ver t1 t2 ...]   (--ver: só quadros de teste)
AJUSTE: QUAD (trapézio do chão) e o bloco "linha do tempo" com as frases e tempos da transcrição."""
import sys, subprocess, os, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1080, 1920, 30
A = "/System/Library/Fonts/Avenir Next.ttc"
C = "/System/Library/Fonts/Avenir Next Condensed.ttc"
F = {"a": (A, 8), "ai": (A, 9), "c": (C, 8), "ci": (C, 9)}

# ---------- chão: retângulo plano LW x LH -> trapézio com fuga em (540,680) ----------
LW, LH = 1600, 1780
QUAD = np.float32([[318, 1080], [762, 1080], [1300, 2050], [-220, 2050]])
M = cv2.getPerspectiveTransform(np.float32([[0, 0], [LW, 0], [LW, LH], [0, LH]]), QUAD)
Minv = np.linalg.inv(M)

def tela_p_chao(x, y):
    v = Minv @ np.array([x, y, 1.0]); return v[0] / v[2], v[1] / v[2]

# ---------- linha do tempo ----------
# item: (camada, texto, fonte, tamanho, x, y, entra, sai, cor)   cor "b" branco | "y" amarelo claro
IT = []
def _larg(t, f, s): return ImageFont.truetype(F[f][0], s, index=F[f][1]).getlength(t)

def largura_util(y, m=50):
    if y is None: return W - 2 * m
    a, _ = tela_p_chao(m, y); b, _ = tela_p_chao(W - m, y)
    return min(b, LW - 60) - max(a, 60)

def linha(camada, palavras, x, y, sai, gap=0.26, maxw=None):
    """palavras: [(texto, fonte, tam, entra, cor)] lado a lado, centradas em (x,y) da tela; encolhe p/ caber"""
    avail = maxw or largura_util(y if camada == "chao" else None)
    tot = lambda k: sum(_larg(t, f, int(s * k)) for t, f, s, _, _ in palavras) + sum(gap * s * k for _, _, s, _, _ in palavras[:-1])
    k = min(1.0, avail / tot(1.0))
    cx, cy = tela_p_chao(x, y) if camada == "chao" else (x, y)
    xx = cx - tot(k) / 2
    for t, f, s, ent, cor in palavras:
        s2 = int(s * k); l = _larg(t, f, s2)
        IT.append((camada, t, f, s2, xx + l / 2, cy, ent, sai, cor))
        xx += l + gap * s2

def fundo(t, f, s, x, y, ent, sai, cor="b", maxw=None):
    linha("fundo", [(t, f, s, ent, cor)], x, y, sai, maxw=maxw)

def chao(t, f, s, x, y, ent, sai, cor="b", maxw=None):
    linha("chao", [(t, f, s, ent, cor)], x, y, sai, maxw=maxw)

P = "a"   # texto corrido: Avenir Next Heavy
# 0–0.95  GANCHO: "DEPOIS DESSE VÍDEO" gigante atrás dela, já no 1º quadro
fundo("DEPOIS", "c", 420, 540, 300, -0.15, 0.95)
fundo("DESSE", "c", 420, 540, 640, 0.20, 0.95)
fundo("VÍDEO,", "c", 420, 540, 985, 0.40, 0.95, "y")
# 1.0–2.2  "você vai ver como é FÁCIL"
linha("chao", [("você", P, 240, 1.00, "b"), ("vai", P, 240, 1.06, "b"), ("ver", P, 240, 1.18, "b")], 540, 1470, 2.18)
linha("chao", [("como", P, 240, 1.30, "b"), ("é", P, 240, 1.52, "b")], 540, 1690, 2.18)
fundo("FÁCIL", "c", 600, 540, 520, 1.62, 2.18, "y")
# 2.18–4.74 "você fazer um vídeo de CONTEÚDO"
linha("chao", [("você", P, 240, 2.18, "b"), ("fazer", P, 240, 2.64, "b"), ("um", P, 240, 3.12, "b")], 540, 1470, 4.74)
linha("chao", [("vídeo", P, 240, 3.64, "b"), ("de", P, 240, 4.04, "b")], 540, 1690, 4.74)
fundo("CONTEÚDO", "c", 600, 540, 540, 4.18, 4.74)
# 4.74–8.4 PICO: chão e fundo lotados, sem sobrepor
FIM = 8.42
linha("chao", [("com", P, 240, 4.74, "b"), ("TEXTO", "c", 420, 5.24, "y")], 540, 1420, FIM)
linha("chao", [("aparecendo", P, 240, 5.72, "b"), ("em", P, 240, 6.40, "b"), ("vários", P, 240, 6.64, "b")], 540, 1600, FIM)
linha("chao", [("LUGARES", "c", 400, 6.98, "y"), ("DIFERENTES.", "c", 400, 7.54, "y")], 540, 1790, FIM)
# fundo: parede de TEXTO cobrindo a tela toda, inclusive atrás dela (ela fica na frente)
import random
random.seed(7)
fileiras = [(110, 3), (330, 2), (560, 3), (790, 2), (1020, 3)]
pos = []
for y, n in fileiras:
    xs = [180, 540, 900] if n == 3 else [275, 805]
    for x in xs: pos.append((x, y, 320 if n == 3 else 450))
ordem = list(range(len(pos))); random.shuffle(ordem)
for k, j in enumerate(ordem):
    x, y, mw = pos[j]
    fundo("TEXTO", "c", 600, x, y, 5.24 + k * 0.17, FIM, "y" if j in (5, 9) else "b", maxw=mw)
# 8.62–11.0 "e tudo sem você manjar de EDIÇÃO"
linha("chao", [("e", P, 240, 8.62, "b"), ("tudo", P, 240, 8.98, "b"), ("sem", P, 240, 9.42, "b"), ("você", P, 240, 9.74, "b")], 540, 1470, 11.02)
linha("chao", [("manjar", P, 240, 10.02, "b"), ("de", P, 240, 10.40, "b")], 540, 1690, 11.02)
fundo("EDIÇÃO.", "c", 600, 540, 540, 10.54, 11.02, "y")
# 11.1–fim "o CLAUDE vai fazer pra você."
END = 16.2
fundo("CLAUDE", "c", 600, 540, 520, 11.30, END, "y")
linha("chao", [("vai", P, 240, 11.58, "b"), ("fazer", P, 240, 12.00, "b")], 540, 1470, END)
linha("chao", [("pra", P, 260, 12.40, "b"), ("você.", P, 260, 12.56, "b")], 540, 1690, END)

# ---------- desenho ----------
_sp = {}
def sprite(t, f, s):
    k = (t, f, s)
    if k not in _sp:
        fo = ImageFont.truetype(F[f][0], s, index=F[f][1])
        l, tp, r, b = fo.getbbox(t)
        im = Image.new("L", (r - l + 40, b - tp + 40), 0)
        ImageDraw.Draw(im).text((20 - l, 20 - tp), t, font=fo, fill=255)
        _sp[k] = np.asarray(im, np.float32) / 255
    return _sp[k]

def anim(t, ent, sai):
    u = (t - ent) / 0.22
    if u <= 0: return 0, 0
    if u < 1:
        c1 = 2.2; e = 1 + (c1 + 1) * (u - 1) ** 3 + c1 * (u - 1) ** 2   # easeOutBack
        esc, al = 0.55 + 0.45 * e, min(1, u * 3)
    else:
        esc, al = 1, 1
    v = (sai - t) / 0.12
    if v <= 0: return 0, 0
    if v < 1: al *= v; esc *= 0.94 + 0.06 * v
    return esc, al

def cola(can, spr, cx, cy, esc, al):
    if esc <= 0.02: return
    h, w = spr.shape
    s = cv2.resize(spr, (max(1, int(w * esc)), max(1, int(h * esc))), interpolation=cv2.INTER_AREA) * al
    h, w = s.shape; x0, y0 = int(cx - w / 2), int(cy - h / 2)
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(can.shape[1], x0 + w), min(can.shape[0], y0 + h)
    if X1 <= X0 or Y1 <= Y0: return
    reg = can[Y0:Y1, X0:X1]
    np.maximum(reg, s[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0], out=reg)

def camadas(t):
    ch = np.zeros((LH, LW), np.float32); fu = np.zeros((H, W), np.float32)
    chy = np.zeros((LH, LW), np.float32); fuy = np.zeros((H, W), np.float32)
    for cam, tx, f, s, x, y, ent, sai, cor in IT:
        esc, al = anim(t, ent, sai)
        if al > 0:
            sp = sprite(tx, f, s)
            cola(ch if cam == "chao" else fu, sp, x, y, esc, al)
            if cor == "y": cola(chy if cam == "chao" else fuy, sp, x, y, esc, al)
    wp = lambda a: cv2.warpPerspective(a, M, (W, H), flags=cv2.INTER_LINEAR)
    return wp(ch), fu, wp(chy), fuy

def compor(f, t, pes):
    ac, af, acy, afy = camadas(t)
    AMA = np.array([255, 226, 110], np.float32)   # amarelo claro
    cor = lambda a, ay, base: base * (1 - (ay / np.maximum(a, 1e-3))[..., None]) + AMA * (ay / np.maximum(a, 1e-3))[..., None]
    livre = (1 - pes)[..., None]
    lum = cv2.GaussianBlur(f.mean(2), (0, 0), 3) / 200
    # fundo: sombra forte (céu claro atrás) + leve contorno
    sh = np.maximum(cv2.GaussianBlur(np.roll(np.roll(af, 10, 0), 6, 1), (0, 0), 11) * 0.7,
                    cv2.GaussianBlur(af, (0, 0), 3) * 0.55)
    f = f * (1 - sh[..., None] * livre)
    f = f * (1 - af[..., None] * livre) + cor(af, afy, np.float32(252)) * af[..., None] * livre
    # chão: sombra suave + branco que pega a luz do chão
    sc = np.maximum(cv2.GaussianBlur(np.roll(ac, 5, 0), (0, 0), 7) * 0.6, cv2.GaussianBlur(ac, (0, 0), 2.5) * 0.45)
    f = f * (1 - sc[..., None] * livre)
    luz = np.clip(lum, 0.78, 1.0)[..., None]
    f = f * (1 - ac[..., None] * livre) + cor(ac, acy, np.float32(250)) * luz * ac[..., None] * livre
    return np.clip(f, 0, 255).astype(np.uint8)

def masc(i):
    p = f"mask_fina/{i:05d}.png" if os.path.exists("mask_fina") else f"mask/{i:05d}.png"
    if not os.path.exists(p): return np.zeros((H, W), np.float32)
    m = cv2.imread(p, 0).astype(np.float32) / 255
    # recorte sólido: fecha buracos (mão na bolsa, punho, tênis) que deixavam o texto vazar
    b = (m > 0.08).astype(np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    fl = b.copy(); cv2.floodFill(fl, np.zeros((H + 2, W + 2), np.uint8), (0, 0), 1)
    b = b | (1 - fl)
    b = cv2.erode(b, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return np.maximum(np.clip(m / 0.45, 0, 1), cv2.GaussianBlur(b.astype(np.float32), (0, 0), 1.5))

if "--ver" in sys.argv:
    for ts in sys.argv[sys.argv.index("--ver") + 1:]:
        t = float(ts); i = int(round(t * FPS)) + 1
        f = cv2.cvtColor(cv2.imread(f"quadros/{i:05d}.jpg"), cv2.COLOR_BGR2RGB).astype(np.float32)
        cv2.imwrite(f"ver_{t:05.2f}.jpg", cv2.cvtColor(compor(f, t, masc(i)), cv2.COLOR_RGB2BGR))
    sys.exit()

pb = subprocess.Popen(["ffmpeg", "-v", "error", "-i", "base.mp4", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
out = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                        "-i", "base.mp4", "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", "comp.mp4"], stdin=subprocess.PIPE)
i = 0
while True:
    b = pb.stdout.read(W * H * 3)
    if len(b) < W * H * 3: break
    i += 1
    f = np.frombuffer(b, np.uint8).reshape(H, W, 3).astype(np.float32)
    out.stdin.write(compor(f, (i - 1) / FPS, masc(i)).tobytes())
out.stdin.close(); out.wait(); print("comp.mp4", i, "quadros")
