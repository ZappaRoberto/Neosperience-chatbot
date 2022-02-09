import requests
import json


def ask_questions():
    if input("Do you want ask questions? [y/n]\n").lower() == "y":
        return True
    else:
        return False


def main():
    invoke_url = "https://66mlkwpffi.execute-api.us-east-1.amazonaws.com/dev"
    data = {
        "string": {
            "inputs": {
                "past_user_inputs": [],
                "generated_responses": [],
                "text": ""
            }
        }
    }

    while ask_questions():
        data["string"]["inputs"].update({"text": input("Write your question\n")})
        response = requests.post(invoke_url, data=json.dumps(data), headers={'Content-type': 'application/json'})
        try:
            # make response a dictionary
            response = response.json()
            # make the response's body a dictionary
            response = json.loads(response["body"])
            # print the model's prediction
            print(response["generated_text"])
            # update my dictionary
            data["string"]["inputs"].update({"past_user_inputs": response["conversation"]["past_user_inputs"],
                                             "generated_responses": response["conversation"]["generated_responses"]})
        except TypeError as tp:
            print("TypeError: ", tp)


if __name__ == "__main__":
    main()
