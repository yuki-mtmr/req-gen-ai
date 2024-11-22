import operator
from typing import Annotated

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from documentation_agent.persona import Persona


class Interview(BaseModel):
    persona: Persona = Field(..., description="インタビュー対象のペルソナ")
    question: str = Field(..., description="インタビューでの質問")
    answer: str = Field(..., description="インタビューでの回答")


class InterviewResult(BaseModel):
    interviews: list[Interview] = Field(
        default_factory=list, description="インタビュー結果のリスト"
    )


class InterviewState(BaseModel):
    user_request: str = Field(..., description="ユーザーからのリクエスト")
    personas: Annotated[list[Persona], operator.add] = Field(
        default_factory=list, description="生成されたペルソナのリスト"
    )
    interviews: Annotated[list[Interview], operator.add] = Field(
        default_factory=list, description="実施されたインタビューのリスト"
    )
    requirements_doc: str = Field(default="", description="生成された要件定義")
    iteration: int = Field(
        default=0, description="ペルソナ生成とインタビューの反復回数"
    )
    is_information_sufficient: bool = Field(
        default=False, description="情報が十分かどうか"
    )


class InterviewConductor:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, user_request: str, personas: list[Persona]) -> InterviewResult:
        # 質問を生成
        questions = self._generate_questions(
            user_request=user_request, personas=personas
        )
        # 回答を生成
        answers = self._generate_answers(
            personas=personas, questions=questions)
        # 質問と回答の組み合わせからインタビューリストを作成
        interviews = self._create_interviews(
            personas=personas, questions=questions, answers=answers
        )
        # インタビュー結果を返す
        return InterviewResult(interviews=interviews)

    def _generate_questions(
        self, user_request: str, personas: list[Persona]
    ) -> list[str]:
        # 質問生成のためのプロンプトを定義
        question_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたはユーザー要件に基づいて適切な質問を生成する専門家です。",
                ),
                (
                    "human",
                    "以下のペルソナに関連するユーザーリクエストについて、1つの質問を生成してください。\n\n"
                    "ユーザーリクエスト: {user_request}\n"
                    "ペルソナ: {persona_name} - {persona_background}\n\n"
                    "質問は具体的で、このペルソナの視点から重要な情報を引き出すように設計してください。",
                ),
            ]
        )
        # 質問生成のためのチェーンを作成
        question_chain = question_prompt | self.llm | StrOutputParser()

        # 各ペルソナに対する質問クエリを作成
        question_queries = [
            {
                "user_request": user_request,
                "persona_name": persona.name,
                "persona_background": persona.background,
            }
            for persona in personas
        ]
        # 質問をバッチ処理で生成
        return question_chain.batch(question_queries)

    def _generate_answers(
        self, personas: list[Persona], questions: list[str]
    ) -> list[str]:
        # 回答生成のためのプロンプトを定義
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたは以下のペルソナとして回答しています: {persona_name} - {persona_background}",
                ),
                ("human", "質問: {question}"),
            ]
        )
        # 回答生成のためのチェーンを作成
        answer_chain = answer_prompt | self.llm | StrOutputParser()

        # 各ペルソナに対する回答クエリを作成
        answer_queries = [
            {
                "persona_name": persona.name,
                "persona_background": persona.background,
                "question": question,
            }
            for persona, question in zip(personas, questions)
        ]
        # 回答をバッチ処理で生成
        return answer_chain.batch(answer_queries)

    def _create_interviews(
        self, personas: list[Persona], questions: list[str], answers: list[str]
    ) -> list[Interview]:
        # ペルソナ毎に質問と回答の組み合わせからインタビューオブジェクトを作成
        return [
            Interview(persona=persona, question=question, answer=answer)
            for persona, question, answer in zip(personas, questions, answers)
        ]
