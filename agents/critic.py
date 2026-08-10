import ast

class CriticAgent:
    def review(self, code: str):
        try:
            ast.parse(code)
            return 'ok', True
        except SyntaxError as e:
            return str(e), False
