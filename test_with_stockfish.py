import chess
import chess.engine
import os
import openai  # openai 패키지 설치 필요: pip install openai

# 1. Stockfish 엔진 열기 (설치 경로에 맞게 수정)
engine = chess.engine.SimpleEngine.popen_uci(["/usr/games/stockfish"])
board = chess.Board()

# 새로운 Python SDK 스타일 (client 객체)
client = openai.OpenAI(
    api_key= ""
)

def ask_gpt_for_move(fen: str) -> str:
    """
    현재 FEN을 GPT에게 넘겨서
    - 합법적인 UCI 수 하나만 출력하도록 요청
    - 그 결과를 uci 문자열로 리턴

    전제:
    - OPENAI_API_KEY는 환경변수나 다른 방식으로 이미 설정되어 있다고 가정
    - 모델 이름은 gpt-4.1-mini 예시 (원하는 걸로 교체 가능)
    """
    system_prompt = (
        "You are a chess engine that only responds with a single legal move "
        "in UCI format (like e2e4, g1f3, e7e8q). "
        "Do not include any explanation or extra text, only the move."
    )

    user_prompt = (
        f"The current chess position is given in FEN:\n\n{fen}\n\n"
        "It is your turn to move. "
        "Respond with exactly one legal move in UCI format, for example: e2e4"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,  # 너무 랜덤하지 않게 약간만
        max_tokens=10,    # 한 수만 출력하게 짧게
    )

    move_uci = response.choices[0].message.content.strip()
    # 혹시 개행이나 공백이 섞여 있을 경우 대비해서 정제
    move_uci = move_uci.split()[0]

    return move_uci.strip()


while not board.is_game_over():
    print(board)
    print("Turn:", "White" if board.turn == chess.WHITE else "Black")

    if board.turn == chess.WHITE:
        # GPT가 White 라고 가정
        move_uci = ask_gpt_for_move(board.fen())
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            print("불법 수입니다. 다시 입력해주세요.")
            continue
        board.push(move)
    else:
        # Stockfish가 Black
        result = engine.play(board, chess.engine.Limit(time=0.5))
        print("Stockfish plays:", result.move.uci())
        board.push(result.move)

print("Game over:", board.result())
engine.quit()
