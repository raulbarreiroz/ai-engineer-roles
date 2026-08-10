class ExecutorAgent:
    def generate(self, instruction: str, plan):
        if 'suma' in instruction.lower():
            return 'def sum_integers(values):\n    return sum(values)\n'
        return 'def fibonacci(n):\n    a,b=0,1\n    for _ in range(n): a,b=b,a+b\n    return a\n'
