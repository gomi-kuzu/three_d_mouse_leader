#!/usr/bin/env python3
"""
特異点・可操作度 リアルタイム可視化 (軽量版)

スライダーによる手動操作 OR ROS2 /joint_states 購読 に対応。
FuncAnimation (10 Hz) で確実にリアルタイム更新する。

実行例:
  # スライダー手動操作モード
  conda run -n leros_jazzy python3 visualize_singularity_rt.py

  # ROS2 リアルタイムモード (別端末で spacemouse_ik.launch.py を起動しておく)
  source ~/jazzy_ws/install/setup.bash
  conda run -n leros_jazzy python3 visualize_singularity_rt.py --ros2
"""
import warnings
warnings.filterwarnings('ignore')

import japanize_matplotlib  # noqa: F401  日本語フォント有効化
import threading
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
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
NDOF  = len(CTRL)
NAMES = ['Pan', 'Lift', 'Elbow', 'W.Flex', 'W.Roll']
LIMITS = [(-90, 90), (-90, 90), (-90, 90), (-90, 90), (-180, 180)]

# ── 共有状態 (スライダー / ROS2 どちらも書き込む) ────────────────────────────
_lock   = threading.Lock()
_q_deg  = np.array([0., -30., 60., 30., 0.], dtype=float)   # 現在の関節角度 [deg]
_source = 'slider'    # 'slider' or 'ros2'

def set_q(q: np.ndarray, src: str = 'slider'):
    global _source
    with _lock:
        _q_deg[:] = q
        _source = src

def get_q() -> tuple[np.ndarray, str]:
    with _lock:
        return _q_deg.copy(), _source

# ── キネマティクス ─────────────────────────────────────────────────────────
def get_jacobian(q_deg: np.ndarray):
    qf = np.zeros(6)
    qf[cidx] = np.deg2rad(q_deg)
    J_raw, T = robot.velocity_control_matrices(jnp.array(qf))
    J_full = np.array(J_raw)
    J = J_full[:, cidx] if J_full.shape[1] > NDOF else J_full
    return J, np.array(T)

def svd_analyze(J: np.ndarray) -> dict:
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    J3 = J[:3, :]
    Up, sp, _ = np.linalg.svd(J3, full_matrices=False)
    w_pos = np.sqrt(max(0., np.linalg.det(J3 @ J3.T)))
    cond = s[0] / s[-1] if s[-1] > 1e-12 else np.inf
    return dict(s=s, Vt=Vt, s_pos=sp, U_pos=Up, w_pos=w_pos, cond=cond)

# ── ROS2 購読スレッド ─────────────────────────────────────────────────────────
def start_ros2(node_args=None):
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState

        rclpy.init(args=node_args)
        node = Node('singularity_viz')

        name2ci = {n: i for i, n in enumerate(CTRL)}

        def cb(msg: JointState):
            q, _ = get_q()           # 現在値をベース (未受信関節は維持)
            for i, name in enumerate(msg.name):
                if name in name2ci:
                    q[name2ci[name]] = float(np.rad2deg(msg.position[i]))
            set_q(q, 'ros2')

        # spacemouse_ik_node が /joint_states に URDF 名 (shoulder_pan 等) で配信
        node.create_subscription(JointState, '/joint_states', cb, 10)
        print('[ROS2] /joint_states 購読開始', flush=True)
        rclpy.spin(node)
        rclpy.shutdown()
    except Exception as e:
        print(f'[ROS2] エラー: {e}', flush=True)

# ── 引数パース ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--ros2', action='store_true', help='ROS2 /joint_states を購読')
args, _ = parser.parse_known_args()

# ── 初期計算 ────────────────────────────────────────────────────────────────
print('初期化中...', flush=True)
q_init = _q_deg.copy()
J0, _ = get_jacobian(q_init)
a0 = svd_analyze(J0)
print('完了!', flush=True)

# ── レイアウト ──────────────────────────────────────────────────────────────
SING_THR = 1e-3
plt.rcParams['font.size'] = 9

fig = plt.figure(figsize=(15, 8))
title_src = 'ROS2 /joint_states' if args.ros2 else 'スライダー手動操作'
fig.suptitle(f'SO-101 特異点・可操作度 リアルタイム可視化  [{title_src}]',
             fontsize=12, fontweight='bold', y=0.97)

slider_bottom = 0.32 if not args.ros2 else 0.05
gs = gridspec.GridSpec(2, 3, fig,
                       top=0.88, bottom=slider_bottom,
                       hspace=0.65, wspace=0.42)

ax_sv   = fig.add_subplot(gs[0, 0])   # 特異値バーチャート
ax_ell  = fig.add_subplot(gs[0, 1])   # 可操作度楕円
ax_ns   = fig.add_subplot(gs[0, 2])   # ヌル空間ベクトル
ax_mtr  = fig.add_subplot(gs[1, 0])   # メトリクス
ax_pos  = fig.add_subplot(gs[1, 1])   # 位置部分特異値
ax_info = fig.add_subplot(gs[1, 2])   # 関節角度 + ソース表示

# ── スライダー (スライダーモード時のみ配置) ────────────────────────────────
sliders = []
if not args.ros2:
    for i in range(NDOF):
        ax_s = fig.add_axes([0.08, 0.020 + i * 0.056, 0.84, 0.030])
        s = Slider(ax_s, f'{NAMES[i]} [deg]', LIMITS[i][0], LIMITS[i][1],
                   valinit=q_init[i], valstep=0.5)
        sliders.append(s)

    def on_slider(_):
        q = np.array([s.val for s in sliders])
        set_q(q, 'slider')

    for s in sliders:
        s.on_changed(on_slider)
else:
    # ROS2 モード: ダミースライダー (値参照用)
    for i in range(NDOF):
        ax_s = fig.add_axes([0, 0, 0.001, 0.001])  # 非表示領域
        s = Slider(ax_s, '', LIMITS[i][0], LIMITS[i][1],
                   valinit=q_init[i])
        sliders.append(s)

# ── ヘルパー ────────────────────────────────────────────────────────────────
def bar_colors(sv):
    return ['#e74c3c' if v < SING_THR
            else '#f39c12' if v < SING_THR * 10
            else '#27ae60'
            for v in sv]

def draw_ellipse(ax, a: dict):
    """可操作度楕円を XZ 面に投影して描画 (毎フレームクリア)"""
    ax.clear()
    sp = a['s_pos'][:3]
    Up = a['U_pos'][:, :3]
    P  = np.array([[1., 0., 0.], [0., 0., 1.]])
    A  = P @ Up @ np.diag(sp)
    Ua2, sa2, _ = np.linalg.svd(A, full_matrices=False)
    theta = np.linspace(0, 2 * np.pi, 200)
    ell   = Ua2 @ np.diag(sa2) @ np.vstack([np.cos(theta), np.sin(theta)])
    sc    = 0.4 / max(sp[0], 1e-8)
    ax.plot(ell[0] * sc, ell[1] * sc, 'b-', lw=1.5, alpha=0.85)
    ax.fill(ell[0] * sc, ell[1] * sc, alpha=0.07, color='blue')
    c3 = ['#e74c3c', '#3498db', '#2ecc71']
    for k in range(3):
        dx, dz = Up[0, k] * sp[k] * sc, Up[2, k] * sp[k] * sc
        ax.annotate('', xy=(dx, dz), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=c3[k], lw=2.0))
        ax.text(dx * 1.3, dz * 1.3, f'σ{k+1}={sp[k]:.3f}',
                fontsize=7.5, color=c3[k], ha='center')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    lim = max(sa2[0] * sc * 1.5, 0.05)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set(title='可操作度楕円 (XZ 面)\n潰れるほど特異点に近い',
           xlabel='X 方向', ylabel='Z 方向')

def draw_nullspace(ax, a: dict):
    """最小右特異ベクトル (毎フレームクリア)"""
    ax.clear()
    v = a['Vt'][-1, :]
    s_min = a['s'][-1]
    ax.bar(range(NDOF), v, color='#3498db', edgecolor='k', lw=0.3)
    ax.axhline(0, color='k', lw=0.5)
    ax.set(title=r'ヌル空間方向 ($\sigma_{min}=' + f'{s_min:.4f}' + r'$)' + '\nこの関節方向に動かしても EE が動かない',
           xticks=range(NDOF), xticklabels=NAMES, ylabel='成分値')
    ax.set_ylim(-1.15, 1.15)
    ax.grid(axis='y', alpha=0.3)

def fmt_metrics(a: dict) -> tuple[str, str]:
    s_min = a['s'][-1]
    if   s_min < 1e-4: col, stat = '#c0392b', '★  特異点!'
    elif s_min < 1e-3: col, stat = '#e74c3c', '▲  危険'
    elif s_min < 1e-2: col, stat = '#f39c12', '△  注意'
    else:              col, stat = '#27ae60', '✓  正常'
    cstr = f"{a['cond']:.1f}" if not np.isinf(a['cond']) else '∞'
    txt = (f"状態:   {stat}\n\n"
           f"w_pos   = {a['w_pos']:.6f}\n"
           f"σ_min   = {a['s'][-1]:.6f}\n"
           f"κ(J)    = {cstr}\n\n"
           "全特異値:\n" +
           '\n'.join(f"  σ{i+1} = {v:.5f}" for i, v in enumerate(a['s'])))
    return txt, col

# ── 初期プロット ─────────────────────────────────────────────────────────────
# [1] 特異値バーチャート
sv_bars = ax_sv.bar(range(NDOF), a0['s'], color=bar_colors(a0['s']),
                    edgecolor='k', lw=0.3)
ax_sv.axhline(SING_THR, color='red', ls='--', lw=1.2, label=f'閾値={SING_THR}')
ax_sv.set(title=r'全特異値 $\sigma_1 \geq \cdots \geq \sigma_5$' + '\n(赤=特異点)',
          xticks=range(NDOF),
          xticklabels=[r'$\sigma_' + str(i+1) + '$' for i in range(NDOF)],
          ylabel=r'$\sigma$ 値')
ax_sv.legend(fontsize=8)
ax_sv.grid(axis='y', alpha=0.3)

# [2] 楕円 (初期)
draw_ellipse(ax_ell, a0)

# [3] ヌル空間 (初期)
draw_nullspace(ax_ns, a0)

# [4] メトリクス
ax_mtr.set_axis_off()
mtr_t = ax_mtr.text(0.05, 0.97, '', transform=ax_mtr.transAxes,
                     va='top', fontsize=9.5)
ax_mtr.set_title('リアルタイム メトリクス', fontsize=10)
txt0, col0 = fmt_metrics(a0)
mtr_t.set_text(txt0)
mtr_t.set_color(col0)

# [5] 位置部分特異値
pos_bars = ax_pos.bar(range(3), a0['s_pos'][:3],
                       color=bar_colors(a0['s_pos'][:3]),
                       edgecolor='k', lw=0.3)
ax_pos.axhline(SING_THR, color='red', ls='--', lw=1.2)
ax_pos.set(title=r'位置部分 $J_{[:3,:]}$ 特異値' + '\n= 楕円の 3 主軸の長さ',
           xticks=range(3), xticklabels=[r'$\sigma_1$', r'$\sigma_2$', r'$\sigma_3$'], ylabel=r'$\sigma$ 値')
ax_pos.grid(axis='y', alpha=0.3)

# [6] 関節角度 + ソース表示
ax_info.set_axis_off()
info_t = ax_info.text(0.05, 0.97, '', transform=ax_info.transAxes,
                       va='top', fontsize=9.5, color='#555')
ax_info.set_title('現在の関節角度', fontsize=10)

# ── FuncAnimation コールバック (10 Hz) ────────────────────────────────────
_prev_q = q_init.copy()

def animate(_):
    global _prev_q
    q, src = get_q()

    # 変化がなければスキップ (CPU 節約)
    if np.allclose(q, _prev_q, atol=0.05):
        return
    _prev_q = q.copy()

    try:
        J, _ = get_jacobian(q)
        a = svd_analyze(J)
    except Exception as e:
        print(f'計算エラー: {e}', flush=True)
        return

    # [1] 特異値バーチャート
    for rect, h, c in zip(sv_bars, a['s'], bar_colors(a['s'])):
        rect.set_height(h)
        rect.set_color(c)
    ax_sv.relim()
    ax_sv.autoscale_view()

    # [2] 楕円
    draw_ellipse(ax_ell, a)

    # [3] ヌル空間
    draw_nullspace(ax_ns, a)

    # [4] メトリクス
    txt, col = fmt_metrics(a)
    mtr_t.set_text(txt)
    mtr_t.set_color(col)

    # [5] 位置部分特異値
    for rect, h, c in zip(pos_bars, a['s_pos'][:3], bar_colors(a['s_pos'][:3])):
        rect.set_height(h)
        rect.set_color(c)
    ax_pos.relim()
    ax_pos.autoscale_view()

    # [6] 関節角度テキスト
    src_label = '[ROS2]' if src == 'ros2' else '[スライダー]'
    lines = f"input source: {src_label}\n\n" + '\n'.join(
        f"  {n:8s}: {v:+7.2f} deg" for n, v in zip(NAMES, q))
    info_t.set_text(lines)


ani = FuncAnimation(fig, animate, interval=100, cache_frame_data=False)

# ── ROS2 スレッド起動 (--ros2 オプション時) ──────────────────────────────────
if args.ros2:
    t = threading.Thread(target=start_ros2, daemon=True)
    t.start()

print('ウィンドウを表示中... (閉じると終了)', flush=True)
plt.show()
