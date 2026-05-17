#!/usr/bin/env python3
"""
特異点・可操作度 インタラクティブ可視化
Singularity & Manipulability Interactive Visualization for SO-101 Arm

可視化内容:
  [左上]  全特異値バーチャート    : σ₁≥...≥σ₅, 赤=0に近い→特異点危険
  [中上]  可操作度楕円 (XZ投影)  : EE が「どの方向に動きやすいか」を楕円で表示
  [右上]  可操作度ヒートマップ   : Pan×Lift の設定空間での可操作度分布
  [左下]  ヌル空間ベクトル        : 最小右特異ベクトル = 特異点で「無駄な動き」方向
  [中下]  リアルタイム メトリクス : w_pos, σ_min, 条件数κ
  [右下]  位置部分の特異値        : J[:3,:] の特異値 = 楕円体の3軸の長さ

操作: 下部スライダーで各関節角度を変更 → 全グラフがリアルタイム更新

実行方法:
  conda run -n leros_jazzy python3 visualize_singularity.py
"""
import warnings
warnings.filterwarnings('ignore')

import japanize_matplotlib  # noqa: F401  日本語フォント有効化
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
import frax
import jax
import jax.numpy as jnp
import pathlib

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platforms', 'cpu')

# ── ロボット設定 ─────────────────────────────────────────────────────────────
try:
    from ament_index_python.packages import get_package_share_directory
    URDF = get_package_share_directory('three_d_mouse_leader') + '/urdf/so101_new_calib.urdf'
except Exception:
    URDF = str(pathlib.Path(__file__).parent.parent / 'urdf' / 'so101_new_calib.urdf')
EE_OFFSET = np.array([
    [-1., 0.,  0., -0.0079  ],
    [ 0., 1.,  0., -0.000218],
    [ 0., 0., -1., -0.098127],
    [ 0., 0.,  0.,  1.      ],
], dtype=np.float64)

robot = frax.Manipulator(URDF, ee_offset=EE_OFFSET)
CTRL  = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
cidx  = [robot.joint_name_to_index[n] for n in CTRL]
NDOF  = len(CTRL)   # 5

# ── キネマティクス計算 ─────────────────────────────────────────────────────
def get_jacobian(q_deg: np.ndarray):
    """5 関節角度 [deg] → (6×5 ヤコビ行列 J, 4×4 同次変換行列 T)"""
    qf = np.zeros(6)
    qf[cidx] = np.deg2rad(q_deg)
    J_raw, T = robot.velocity_control_matrices(jnp.array(qf))
    J_full = np.array(J_raw)
    # frax が全関節分の J を返す場合は制御関節の列だけ抽出
    J = J_full[:, cidx] if J_full.shape[1] > NDOF else J_full
    return J, np.array(T)


def svd_analyze(J: np.ndarray) -> dict:
    """
    SVD による可操作度解析.

    Returns
    -------
    s      : (NDOF,)  全特異値 (降順)
    Vt     : (NDOF, NDOF)  右特異ベクトル行列
    s_pos  : (3,)     位置部分 J[:3,:] の特異値
    U_pos  : (3, 3)   位置部分の左特異ベクトル
    w_pos  : float    Yoshikawa 可操作度 (位置のみ)
    cond   : float    条件数 κ(J) = σ₁/σ₅
    """
    U, s, Vt = np.linalg.svd(J, full_matrices=False)

    J3 = J[:3, :]                                          # 位置部分 (3×5)
    Up, sp, _ = np.linalg.svd(J3, full_matrices=False)    # sp: (3,)
    w_pos = np.sqrt(max(0., np.linalg.det(J3 @ J3.T)))    # Yoshikawa 可操作度

    cond = s[0] / s[-1] if s[-1] > 1e-12 else np.inf

    return dict(s=s, Vt=Vt, s_pos=sp, U_pos=Up, w_pos=w_pos, cond=cond)


# ── 2D ヒートマップ事前計算 ─────────────────────────────────────────────────
N_MAP  = 40
pan_v  = np.linspace(-80, 80, N_MAP)
lift_v = np.linspace(-80, 80, N_MAP)
Q_BASE = np.array([0., -30., 60., 30., 0.])

print("ヒートマップを計算中... (数秒かかります)")
hmap_w = np.zeros((N_MAP, N_MAP))
for i, p in enumerate(pan_v):
    for j, l in enumerate(lift_v):
        q = Q_BASE.copy(); q[0] = p; q[1] = l
        try:
            hmap_w[j, i] = svd_analyze(get_jacobian(q)[0])['w_pos']
        except Exception:
            hmap_w[j, i] = 0.
print("完了!\n")

# ── レイアウト ───────────────────────────────────────────────────────────────
SING_THR = 1e-3   # この値以下で「特異点警告」

plt.rcParams['font.size'] = 9
fig = plt.figure(figsize=(17, 10))
fig.suptitle('SO-101 特異点・可操作度 インタラクティブ可視化',
             fontsize=13, fontweight='bold', y=0.99)

gs = gridspec.GridSpec(2, 3, fig, top=0.94, bottom=0.34,
                       hspace=0.55, wspace=0.42)
ax_sv   = fig.add_subplot(gs[0, 0])   # 全特異値
ax_ell  = fig.add_subplot(gs[0, 1])   # 可操作度楕円
ax_hmap = fig.add_subplot(gs[0, 2])   # 2D ヒートマップ
ax_ns   = fig.add_subplot(gs[1, 0])   # ヌル空間ベクトル
ax_mtr  = fig.add_subplot(gs[1, 1])   # メトリクス
ax_pos  = fig.add_subplot(gs[1, 2])   # 位置部分特異値

# スライダー
LIMITS = [(-90, 90), (-90, 90), (-90, 90), (-90, 90), (-180, 180)]
NAMES  = ['Pan', 'Lift', 'Elbow', 'W.Flex', 'W.Roll']
sliders = []
for i in range(NDOF):
    ax_s = fig.add_axes([0.08, 0.018 + i * 0.058, 0.84, 0.030])
    s = Slider(ax_s, f'{NAMES[i]} [deg]', LIMITS[i][0], LIMITS[i][1],
               valinit=Q_BASE[i], valstep=0.5)
    sliders.append(s)

# ── ヘルパー ────────────────────────────────────────────────────────────────
def bar_colors(sv):
    """特異値の大きさに応じた棒グラフ色 (赤=危険, 橙=注意, 緑=正常)"""
    return ['#e74c3c' if v < SING_THR
            else '#f39c12' if v < SING_THR * 10
            else '#27ae60'
            for v in sv]


def draw_ellipse(ax, a: dict):
    """
    位置部分可操作度楕円を XZ 面に投影して描画.

    3D 楕円体 E = {Up @ diag(sp) @ w : |w|=1} を XZ 平面へ射影した
    2D 楕円を描画する。楕円が潰れるほど特異点に近い。
    """
    ax.clear()
    sp = a['s_pos'][:3]
    Up = a['U_pos'][:, :3]

    # XZ 射影行列 (x 成分=行0, z 成分=行2)
    P  = np.array([[1., 0., 0.], [0., 0., 1.]])
    A  = P @ Up @ np.diag(sp)           # 2×3 → 投影後の楕円行列
    Ua2, sa2, _ = np.linalg.svd(A, full_matrices=False)  # 2D楕円の主軸

    theta = np.linspace(0, 2 * np.pi, 300)
    ell   = Ua2 @ np.diag(sa2) @ np.vstack([np.cos(theta), np.sin(theta)])

    # 最大半径で正規化 (表示の一貫性)
    sc = 0.40 / max(sp[0], 1e-8)

    ax.plot(ell[0] * sc, ell[1] * sc, 'b-', lw=1.5, alpha=0.85)
    ax.fill(ell[0] * sc, ell[1] * sc, alpha=0.07, color='blue')

    colors3 = ['#e74c3c', '#3498db', '#2ecc71']
    for k in range(3):
        dx = Up[0, k] * sp[k] * sc
        dz = Up[2, k] * sp[k] * sc
        ax.annotate('', xy=(dx, dz), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=colors3[k], lw=2.0))
        ax.text(dx * 1.25, dz * 1.25, f'σ{k+1}={sp[k]:.3f}',
                fontsize=7.5, color=colors3[k], ha='center')

    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    lim = max(sa2[0] * sc * 1.5, 0.05) if len(sa2) > 0 and sa2[0] > 0 else 0.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set(title='可操作度楕円 (XZ 面投影)\n矢印長さ ∝ 特異値 = その方向の動きやすさ',
           xlabel='X 方向速度能力', ylabel='Z 方向速度能力')


def draw_nullspace(ax, a: dict):
    """
    最小右特異ベクトル V_min の棒グラフ.

    特異点付近では σ_min ≈ 0 となり、対応する右特異ベクトル (joint velocity 方向)
    が「ヌル空間方向」= この方向に関節を動かしても EE はほとんど動かない。
    棒の大きい関節ほど「無駄に動く」関節。
    """
    ax.clear()
    v_null = a['Vt'][-1, :]
    s_min  = a['s'][-1]
    ax.bar(range(NDOF), v_null, color='#3498db', edgecolor='k', lw=0.3)
    ax.axhline(0, color='k', lw=0.5)
    ax.set(title=(f'最小右特異ベクトル V_min  (σ_min = {s_min:.5f})\n'
                  '← 特異点付近でこの方向に動かしても EE が動かない'),
           xticks=range(NDOF), xticklabels=NAMES, ylabel='成分値')
    ax.set_ylim(-1.15, 1.15)
    ax.grid(axis='y', alpha=0.3)


def fmt_metrics(a: dict):
    """メトリクステキストと色を返す"""
    s_min = a['s'][-1]
    if s_min < 1e-4:
        col, stat = '#c0392b', '★  特異点!'
    elif s_min < 1e-3:
        col, stat = '#e74c3c', '▲  危険'
    elif s_min < 1e-2:
        col, stat = '#f39c12', '△  注意'
    else:
        col, stat = '#27ae60', '✓  正常'

    cstr = f"{a['cond']:.1f}" if not np.isinf(a['cond']) else '∞'
    lines = [
        f'状態:   {stat}',
        '',
        f'w_pos   = {a["w_pos"]:.6f}',
        f'  (Yoshikawa 可操作度 ≥0,',
        f'   0 で完全特異点)',
        '',
        f'σ_min   = {a["s"][-1]:.6f}',
        f'κ(J)    = {cstr}',
        f'  (条件数: 大きいほど危険)',
        '',
        '全特異値:',
    ] + [f'  σ{i+1} = {v:.5f}' for i, v in enumerate(a['s'])]
    return '\n'.join(lines), col


# ── 初期描画 ─────────────────────────────────────────────────────────────────
J0, T0 = get_jacobian(Q_BASE)
a0 = svd_analyze(J0)

# [1] 全特異値バーチャート
sv_bars = ax_sv.bar(range(NDOF), a0['s'], color=bar_colors(a0['s']),
                    edgecolor='k', lw=0.3)
ax_sv.axhline(SING_THR, color='red', ls='--', lw=1.2,
              label=f'閾値={SING_THR}')
ax_sv.set(title='全特異値 σ₁ ≥ σ₂ ≥ ... ≥ σ₅\n赤 = 特異点に近い',
          xticks=range(NDOF),
          xticklabels=[f'σ{i+1}' for i in range(NDOF)],
          ylabel='σ 値')
ax_sv.legend(fontsize=8)
ax_sv.grid(axis='y', alpha=0.3)

# [2] 可操作度楕円
draw_ellipse(ax_ell, a0)

# [3] ヒートマップ (固定背景 + 動くマーカー)
hm_im  = ax_hmap.imshow(hmap_w, extent=[-80, 80, -80, 80],
                         origin='lower', aspect='auto',
                         cmap='RdYlGn', interpolation='bilinear')
plt.colorbar(hm_im, ax=ax_hmap, label='w_pos', shrink=0.85)
ax_hmap.contour(pan_v, lift_v, hmap_w,
                levels=[5e-5, 5e-4, 5e-3],
                colors=['#c0392b', '#e74c3c', '#f39c12'],
                linewidths=[2.0, 1.5, 1.0])
hm_dot, = ax_hmap.plot(Q_BASE[0], Q_BASE[1], 'b*', ms=12,
                        zorder=5, label='現在位置')
ax_hmap.set(title='可操作度マップ (Pan vs Lift)\n赤等高線 = 特異点近傍領域',
            xlabel='Pan [deg]', ylabel='Lift [deg]')
ax_hmap.legend(fontsize=8)

# [4] ヌル空間ベクトル
draw_nullspace(ax_ns, a0)

# [5] メトリクス
ax_mtr.set_axis_off()
mtr_t = ax_mtr.text(0.05, 0.97, '', transform=ax_mtr.transAxes,
                     va='top', fontsize=9.5)
ax_mtr.set_title('リアルタイム メトリクス', fontsize=10)
txt0, col0 = fmt_metrics(a0)
mtr_t.set_text(txt0)
mtr_t.set_color(col0)

# [6] 位置部分特異値
pos_bars = ax_pos.bar(range(3), a0['s_pos'][:3],
                       color=bar_colors(a0['s_pos'][:3]),
                       edgecolor='k', lw=0.3)
ax_pos.axhline(SING_THR, color='red', ls='--', lw=1.2)
ax_pos.set(title='位置部分 J[:3,:] の特異値\n= 可操作度楕円体の 3 主軸の長さ',
           xticks=range(3), xticklabels=['σ₁', 'σ₂', 'σ₃'], ylabel='σ 値')
ax_pos.grid(axis='y', alpha=0.3)

# ── コールバック ─────────────────────────────────────────────────────────────
def on_change(_):
    q = np.array([s.val for s in sliders])
    try:
        J, _ = get_jacobian(q)
        a = svd_analyze(J)
    except Exception as e:
        print(f'計算エラー: {e}')
        return

    # 全特異値
    for rect, h, c in zip(sv_bars, a['s'], bar_colors(a['s'])):
        rect.set_height(h)
        rect.set_color(c)
    ax_sv.relim()
    ax_sv.autoscale_view()

    # 位置部分特異値
    for rect, h, c in zip(pos_bars, a['s_pos'][:3], bar_colors(a['s_pos'][:3])):
        rect.set_height(h)
        rect.set_color(c)
    ax_pos.relim()
    ax_pos.autoscale_view()

    # 楕円・ヌル空間・メトリクス
    draw_ellipse(ax_ell, a)
    draw_nullspace(ax_ns, a)
    txt, col = fmt_metrics(a)
    mtr_t.set_text(txt)
    mtr_t.set_color(col)

    # ヒートマップ現在位置マーカー
    hm_dot.set_data([q[0]], [q[1]])

    fig.canvas.draw_idle()


for s in sliders:
    s.on_changed(on_change)

plt.show()
