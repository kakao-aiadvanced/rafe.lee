import os

from ollama import Client


class TestService:
    def __init__(self):
        self.client = Client(host="http://localhost:11434")
        self.model = os.getenv("LLM_MODEL", "llama3")

    def router(self, dataset: str) -> str:
        response = self.client.generate(
            model=self.model,
            template="""{{ if .System }}<|start_header_id|>system<|end_header_id|>{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>{{ .Response }}<|eot_id|>""",
            system="You have to answer which teacher can provide better answer. Your answer must one of \"history teacher\" or \"math teacher\". You have to answer correctly.",
            prompt=dataset,
            options={
                "temperature": 0.5,
            },
        )
        return response["response"]

    def qna(self, role, query):
        print('ask to %s' % role)
        if 'math' in role:
            system = "You are a math teacher. answer correctly to question. solve the math problem step by step with detailed explanation."
        elif 'history' in role:
            system = "You are a history teacher. answer correctly to question. explain the historical event with detailed explanation."
        response = self.client.generate(
            model=self.model,
            template="""{{ if .System }}<|start_header_id|>system<|end_header_id|>{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>{{ .Response }}<|eot_id|>""",
            system=system,
            prompt=query,
            options={
                "temperature": 0.5,
            },
        )
        return response["response"]


if __name__ == "__main__":
    service = TestService()
    # query = '235 * 13 + 1234 - 2345'
    query = "why world war 2 started?"
    if 'math' in service.router(query).lower():
        print(service.qna("math teacher", query))
    else:
        print(service.qna("history teacher", query))

