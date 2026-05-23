import sys
from agents.orchestrator import route_question

if __name__ == "__main__":
    question = sys.argv[1]
    answer = route_question(question)
    print(answer)
