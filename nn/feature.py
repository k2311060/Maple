"""ニューラルネットワークの入力特徴生成処理
"""
import numpy as np

from board.constant import PASS
from board.go_board import GoBoard
from board.stone import Stone


def generate_input_planes(board: GoBoard, color: Stone, sym: int=0) -> np.ndarray:
    """ニューラルネットワークの入力データを生成する。

    Args:
        board (GoBoard): 碁盤の情報。
        color (Stone): 手番の色。
        sym (int, optional): 対称形の指定. Defaults to 0.

    Returns:
        numpy.ndarray: ニューラルネットワークの入力データ。
    """
    board_data = board.get_board_data(sym)
    board_size = board.get_board_size()
    # 手番が白の時は石の色を反転する.
    if color is Stone.WHITE:
        board_data = [datum if datum == 0 else (3 - datum) for datum in board_data]

    # 碁盤の各交点の状態
    #     空点 : 1枚目の入力面
    #     自分の石 : 2枚目の入力面
    #     相手の石 : 3枚目の入力面
    board_plane = np.identity(3)[board_data].transpose()

    # n+1手前の着手を取得(n=0,1,2,3,4)
    prev_moves = [PASS if board.moves < i + 1 else board.record.get(board.moves - i - 1) for i in range(5)]

    # n+1手前の着手の座標
    #     着手 : 4+2n枚目の入力面
    #     パス : 5+2n枚目の入力面
    history_planes = [np.zeros(shape=(1, board_size ** 2)) for i in range(5)]
    pass_planes = [np.zeros(shape=(1, board_size ** 2)) for i in range(5)]
    for i in range(5):
        if board.moves > i + 1 and prev_moves[i] == PASS:
            pass_planes[i] = np.ones(shape=(1, board_size ** 2))
        else:
            prev_move_data = [1 if prev_moves[i] == board.get_symmetrical_coordinate(pos, sym) else 0 for pos in board.onboard_pos]
            history_planes[i] = np.array(prev_move_data).reshape(1, board_size ** 2)

    # 手番の色 (14番目の入力面)
    # 黒番は1、白番は-1
    color_plane = np.ones(shape=(1, board_size**2))
    if color == Stone.WHITE:
        color_plane = color_plane * -1

    input_data = np.concatenate([board_plane, history_planes[0], pass_planes[0], history_planes[1], pass_planes[1], history_planes[2], pass_planes[2], history_planes[3], pass_planes[3], history_planes[4], pass_planes[4], color_plane]) \
        .reshape(14, board_size, board_size).astype(np.float32) # pylint: disable=E1121

    return input_data


def generate_target_data(board:GoBoard, target_pos: int, sym: int=0) -> np.ndarray:
    """教師あり学習で使用するターゲットデータを生成する。

    Args:
        board (GoBoard): 碁盤の情報。
        target_pos (int): 教師データの着手の座標。
        sym (int, optional): 対称系の指定。値の範囲は0〜7の整数。デフォルトは0。

    Returns:
        np.ndarray: Policyのターゲットラベル。
    """
    target = [1 if target_pos == board.get_symmetrical_coordinate(pos, sym) else 0 \
        for pos in board.onboard_pos]
    # パスだけ対称形から外れた末尾に挿入する。
    target.append(1 if target_pos == PASS else 0)
    #target_index = np.where(np.array(target) > 0)
    #return target_index[0]
    return np.array(target)


def generate_rl_target_data(board: GoBoard, improved_policy_data: str, sym: int=0) -> np.ndarray:
    """Gumbel AlphaZero方式の強化学習で使用するターゲットデータを精鋭する。

    Args:
        board (GoBoard): 碁盤の情報。
        improved_policy_data (str): Improved Policyのデータをまとめた文字列。
        sym (int, optional): 対称系の指定。値の範囲は0〜7の整数。デフォルトは0。

    Returns:
        np.ndarray: Policyのターゲットデータ。
    """
    split_data = improved_policy_data.split(" ")[1:]
    target_data = [1e-18] * len(board.board)

    for datum in split_data:
        pos, target = datum.split(":")
        coord = board.coordinate.convert_from_gtp_format(pos)
        target_data[coord] = float(target)

    target = [target_data[board.get_symmetrical_coordinate(pos, sym)] for pos in board.onboard_pos]
    target.append(target_data[PASS])

    return np.array(target)
