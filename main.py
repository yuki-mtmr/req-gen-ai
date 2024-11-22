from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from documentation_agent.agent import DocumentationAgent

# .envファイルから環境変数を読み込む
load_dotenv()


# 実行方法:
# poetry run python -m documentation_agent.main --task "ユーザーリクエストをここに入力してください"
# 実行例）
# poetry run python -m documentation_agent.main --task "スマートフォン向けの健康管理アプリを開発したい"
def main():
    import argparse

    # コマンドライン引数のパーサーを作成
    parser = argparse.ArgumentParser(
        description="ユーザー要求に基づいて要件定義を生成します"
    )
    # "task"引数を追加
    parser.add_argument(
        "--task",
        type=str,
        help="作成したいアプリケーションについて記載してください",
    )
    # "k"引数を追加
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="生成するペルソナの人数を設定してください（デフォルト:5）",
    )
    # コマンドライン引数を解析
    args = parser.parse_args()

    # ChatOpenAIモデルを初期化
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    # 要件定義書生成AIエージェントを初期化
    agent = DocumentationAgent(llm=llm, k=args.k)
    # エージェントを実行して最終的な出力を取得
    final_output = agent.run(user_request=args.task)

    # 最終的な出力を表示
    print(final_output)


if __name__ == "__main__":
    main()
