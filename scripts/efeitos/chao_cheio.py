"""CHÃO CHEIO (ótimo pro fechamento/CTA): chão inteiro coberto de fileiras de TEXTO (deslizando, sentidos
alternados) que surgem em cascata, + chamada grande pintada no chão. A pessoa anda por cima.
uso (na pasta da cena): python3 <skill>/scripts/efeitos/chao_cheio.py [--ver t1 t2 ...]   (--ver: só quadros de teste)
AJUSTE: VX, VY (ponto de fuga do chão), F0, polígono CHAO (onde é chão no quadro 1), a palavra das fileiras
(sprite("TEXTO")), o instante em que surgem (0.9 s) e LINHAS (chamada: texto, Z perto, Z longe, largura, cor)."""
import sys, subprocess, os, json, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1080, 1920, 30
C = "/System/Library/Fonts/Avenir Next Condensed.ttc"
AMA = np.array([255, 226, 110], np.float32); BRA = np.array([251, 250, 245], np.float32)
HS = [np.array(h, np.float64) for h in json.load(open("homografias.json"))]
N = len(HS)

# ---------- chão (quadro 1): ponto de fuga (553,720), f=1550, altura 1 ----------
VX, VY, F0 = 553.0, 720.0, 1550.0
def proj(X, Z): return np.stack([VX + F0 * X / Z, VY + F0 / Z], -1)
CHAO = np.zeros((H, W), np.uint8)
cv2.fillPoly(CHAO, [np.int32([[0, 1290], [150, 1140], [300, 1000], [380, 905], [430, 800], [445, 765],
                              [760, 782], [1080, 900], [1080, 1920], [0, 1920]])], 1)
CHAO = cv2.GaussianBlur(CHAO.astype(np.float32), (0, 0), 2)

def fonte(px): return ImageFont.truetype(C, px, index=8)
def sprite(t, px):
    fo = fonte(px); l, tp, r, b = fo.getbbox(t)
    im = Image.new("L", (r - l + 8, b - tp + 8)); ImageDraw.Draw(im).text((4 - l, 4 - tp), t, font=fo, fill=255)
    return np.asarray(im, np.float32) / 255

# ---------- fileiras de TEXTO (mapa em supersample 2x) ----------
SS = 2
yy, xx = np.mgrid[0:H * SS, 0:W * SS].astype(np.float32) / SS
abaixo = yy > VY + 25
ZZ = np.where(abaixo, F0 / np.maximum(yy - VY, 1), 99).astype(np.float32)
XX = ((xx - VX) * ZZ / F0).astype(np.float32)
del yy, xx
Z0, RZ, GZ, PX = 1.22, 0.27, 0.10, 0.50        # 1ª fileira, comprimento da letra no chão, folga, passo em X
PER = RZ + GZ
TS = sprite("TEXTO", 240); TH, TW = TS.shape
TW2 = int(TW * 1.3)                              # tile com espaço
TILE = np.zeros((TH, TW2), np.float32); TILE[:, :TW] = TS
RI = np.floor((ZZ - Z0) / PER).astype(np.int32)
VV = ((RZ - (ZZ - Z0 - RI * PER)) / RZ * TH).astype(np.float32)   # topo da letra = longe
NR = int((8.0 - Z0) / PER) + 1
def fileiras(t, tirar=None):
    if t < 0.88: return None, None
    sent = np.where(RI % 2 == 0, 1.0, -1.0)
    off = (RI % 2) * PX / 2 + sent * 0.10 * (t - 0.9)
    u = (XX - off) / PX
    k = np.floor(u).astype(np.int32)
    U = ((u - k) * TW2).astype(np.float32)
    a = cv2.remap(TILE, U, VV, cv2.INTER_LINEAR, borderValue=0)
    a = np.where((VV >= 0) & (VV < TH) & (RI >= 0) & (RI < NR) & abaixo, a, 0)
    # cascata: fileira mais perto primeiro
    ta = 0.92 + 0.03 * RI
    a = a * np.clip((t - ta) / 0.12, 0, 1)
    a = a * np.clip((8.0 - ZZ) / 2.0, 0, 1)          # some suave lá no fundo
    if tirar is not None:   # tira a palavra inteira que encosta na chamada
        tid = RI * 4096 + (k + 2048)
        ruim = np.unique(tid[(cv2.resize(tirar, (W * SS, H * SS)) > 0.02) & (a > 0.01)])
        a = np.where(np.isin(tid, ruim), 0, a)
    ama = ((k * 7 + RI * 3) % 6 == 0).astype(np.float32)        # algumas em amarelo
    ay = a * ama
    return cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA), cv2.resize(ay, (W, H), interpolation=cv2.INTER_AREA)

# ---------- chamada: linhas pintadas no chão (texto, Zperto, Zlonge, largura X, cor, entra) ----------
TR = json.load(open("transcricao.json")); TR = [w for s in TR for w in s["w"]]
t_ = lambda j: TR[j]["s"]
LINHAS = [("PRA PEGAR ESSE SKILL", 3.95, 4.55, 1.55, "b", t_(2)),
          ("E COLOCAR O CLAUDE", 3.25, 3.78, 1.45, "b", t_(6)),
          ("PRA TRABALHAR PRA VOCÊ.", 2.62, 3.08, 1.45, "b", t_(10)),
          ("COMENTA", 1.98, 2.48, 1.05, "b", t_(0)),
          ("TEXTO", 1.30, 1.88, 0.98, "y", t_(1))]
AMARELAS = {"SKILL", "CLAUDE"}
CHAM = []
for txt, zn, zf, wx, cor, ent in LINHAS:
    fo = fonte(300); pal = txt.split(" "); esp = fo.getlength(" ") * 0.8
    larg = [fo.getbbox(p)[2] - fo.getbbox(p)[0] for p in pal]
    tot = sum(larg) + esp * (len(pal) - 1); l, tp, r, b = fo.getbbox(txt)
    hh = b - tp + 20
    a = Image.new("L", (int(tot) + 20, hh)); ay = Image.new("L", a.size)
    x = 10; tempos = []
    for j, p in enumerate(pal):
        pl, ptp, _, _ = fo.getbbox(p)
        ImageDraw.Draw(a).text((x - pl, 10 - tp), p, font=fo, fill=255)
        if cor == "y" or p.strip(".,") in AMARELAS: ImageDraw.Draw(ay).text((x - pl, 10 - tp), p, font=fo, fill=255)
        tempos.append((int(x), int(x + larg[j]) + 1)); x += larg[j] + esp
    a = np.asarray(a, np.float32) / 255; ay = np.asarray(ay, np.float32) / 255
    sw, sh = a.shape[1], a.shape[0]
    src = np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]])
    X0 = (540 - VX) / F0 * (zn + zf) / 2
    dst = proj(np.array([X0 - wx / 2, X0 + wx / 2, X0 + wx / 2, X0 - wx / 2]), np.array([zf, zf, zn, zn])).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    # palavra a palavra, no tempo da fala
    j0 = [k for k, w in enumerate(TR) if abs(w["s"] - ent) < 1e-6][0]
    CHAM.append(dict(a=a, ay=ay, M=M, faixas=[(x0, x1, TR[j0 + k]["s"]) for k, (x0, x1) in enumerate(tempos)]))

def chamada(t):
    A = np.zeros((H, W), np.float32); AY = np.zeros_like(A)
    for c in CHAM:
        al = np.zeros_like(c["a"])
        for x0, x1, s in c["faixas"]:
            al[:, x0:x1] = np.clip((t - s) / 0.12, 0, 1)
        if al.max() == 0: continue
        A = np.maximum(A, cv2.warpPerspective(c["a"] * al, c["M"], (W, H)))
        AY = np.maximum(AY, cv2.warpPerspective(c["ay"] * al, c["M"], (W, H)))
    return A, AY

def masc(i):
    m = cv2.imread(f"mask/{min(max(i, 1), N):05d}.png", 0).astype(np.float32) / 255
    b = (m > 0.08).astype(np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    fl = b.copy(); cv2.floodFill(fl, np.zeros((H + 2, W + 2), np.uint8), (0, 0), 1)
    b = b | (1 - fl)
    b = cv2.erode(b, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return np.maximum(np.clip(m / 0.45, 0, 1), cv2.GaussianBlur(b.astype(np.float32), (0, 0), 1.5))

AFIM = None
def camada(t):
    global AFIM
    if AFIM is None: AFIM = chamada(99)[0]      # espaço da chamada inteira (reservado desde o início)
    A, AY = chamada(t)
    fa, fy = fileiras(t)
    if fa is not None:   # fileiras viram textura do chão; a chamada abre um respiro justo em volta das letras
        livre = 1 - np.clip(cv2.GaussianBlur(cv2.dilate(A, np.ones((19, 19), np.uint8)), (0, 0), 4) * 1.3, 0, 1)
        fa, fy = fa * livre * 0.62, fy * livre * 0.62
    else:
        fa = fy = np.zeros_like(A)
    return A, AY, fa, fy

def compor(f, t, i):
    A, AY, FA, FY = camada(t)
    Hm = HS[i - 1]; wp = lambda c: cv2.warpPerspective(c, Hm, (W, H)) * cv2.warpPerspective(CHAO, Hm, (W, H))
    A, AY, FA, FY = wp(A), wp(AY), wp(FA), wp(FY)
    livre = (1 - masc(i))
    lum = cv2.GaussianBlur(f.mean(2), (0, 0), 4)
    luz = np.clip(lum / np.percentile(lum[1300:1900], 75), 0.6, 1.04)[..., None]
    for a, ay, forte in [(FA, FY, 0.18), (A, AY, 0.75)]:
        sh = np.maximum(cv2.GaussianBlur(np.roll(a, 6, 0), (0, 0), 7) * forte, cv2.GaussianBlur(a, (0, 0), 2.5) * forte * 0.8)
        f = f * (1 - (sh * livre)[..., None])
        r = (ay / np.maximum(a, 1e-3))[..., None]
        cor = (BRA * (1 - r) + AMA * r) * luz
        k = (a * livre)[..., None]
        f = f * (1 - k) + cor * k
    return np.clip(f, 0, 255).astype(np.uint8)

if "--ver" in sys.argv:
    for ts in sys.argv[sys.argv.index("--ver") + 1:]:
        t = float(ts); i = min(int(round(t * FPS)) + 1, N)
        f = cv2.cvtColor(cv2.imread(f"quadros/{i:05d}.jpg"), cv2.COLOR_BGR2RGB).astype(np.float32)
        cv2.imwrite(f"ver_{t:05.2f}.jpg", cv2.cvtColor(compor(f, t, i), cv2.COLOR_RGB2BGR))
    sys.exit()

pb = subprocess.Popen(["ffmpeg", "-v", "error", "-i", "base.mp4", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
out = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                        "-i", "base.mp4", "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", "comp.mp4"], stdin=subprocess.PIPE)
i = 0
while True:
    b = pb.stdout.read(W * H * 3)
    if len(b) < W * H * 3 or i >= N: break
    i += 1
    f = np.frombuffer(b, np.uint8).reshape(H, W, 3).astype(np.float32)
    out.stdin.write(compor(f, (i - 1) / FPS, i).tobytes())
out.stdin.close(); out.wait(); print("comp.mp4", i, "quadros")
