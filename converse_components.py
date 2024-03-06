from xai_components.base import InArg, OutArg, InCompArg, Component, BaseComponent, xai_component, SubGraphExecutor
import os
import shutil
import subprocess
import glob
import time
import secrets
import random
import string
import json

from pathlib import Path
from flask import Flask, Response, request, jsonify, redirect, render_template, session, abort, send_file
from flask_cors import CORS


alphabet = string.ascii_letters + string.digits


def random_string(length):
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))

def make_id():
    return ''.join(secrets.choice(alphabet) for i in range(29))


@xai_component
class ConverseMakeServer(Component):
    secret_key: InArg[str]
    auth_token: InArg[str]

    def execute(self, ctx) -> None:
        app = Flask(
            'converse', 
            static_folder="public",
            static_url_path=""
        )
        CORS(app)
        app.secret_key = self.secret_key.value if self.secret_key.value is not None else 'opensesame'
        app.config['auth_token'] = self.auth_token.value

        @app.route('/', methods=['GET'])
        def index():
            return redirect('/technologic/index.html')

        @app.route('/technologic/*', methods=['GET'])
        def technologic():
            return send_file('public/technologic/index.html')

        @app.route('/technologic/settings/*', methods=['GET'])
        def settings():
            return send_file('public/technologic/index.html')

        @app.route('/technologic/settings/backends', methods=['GET'])
        def backends():
            return send_file('public/technologic/index.html')

        @app.route('/technologic/settings/backup', methods=['GET'])
        def backup():
            return send_file('public/technologic/index.html')

        @app.route('/technologic/new', methods=['GET'])
        def new_chat():
            return send_file('public/technologic/index.html')

        @app.route('/technologic/*', methods=['GET'])
        def chat():
            return send_file('public/technologic/index.html')


        ctx['flask_app'] = app


@xai_component
class ConverseRun(Component):
    debug_mode: InArg[bool]
    def execute(self, ctx) -> None:
        app = ctx['flask_app']
        # Can't run debug mode from inside jupyter.
        app.run(
            debug=self.debug_mode.value if self.debug_mode.value is not None else True,
            host="0.0.0.0", 
            port=8080
        )



@xai_component
class ConverseDefineAgent(Component):
    on_message: BaseComponent

    name: InCompArg[str]
    message: OutArg[str]
    conversation: OutArg[list]

    def execute(self, ctx) -> None:
        app = ctx['flask_app']

        ctx['converse_model_name'] = self.name.value

        ctx_name = random_string(8)
        self_name = random_string(8)
        fn_name = random_string(8)
        code = f"""
ctx_{ctx_name} = ctx
self_{self_name} = self
@app.route('/chat/completions', methods=['POST'])
def post_route_fn_{fn_name}():
    global ctx_{ctx_name}
    global self_{self_name}

    self = self_{self_name}
    ctx = ctx_{ctx_name}
    app = ctx['flask_app']

    with app.app_context():
        if app.config['auth_token']:
            token = request.headers.get('Authorization')
            if token.split(" ")[1] != app.config['auth_token']:
                abort(401)

    ctx['flask_res'] = None

    data = request.get_json()
    model_name = data['model']
    if model_name != self.name.value:
        abort("model not found")

    messages = data.get('messages', [])
    last_user_message = next((m['content'] for m in reversed(messages) if m['role'] == 'user'), None)

    self.message.value = last_user_message
    self.conversation.value = messages

    stream = data.get('stream', False)

    if stream:
        def stream():
            print("calling on_message")
            next_component = self_{self_name}.on_message.comp
            while next_component:
                next_component = next_component.do(ctx)

                # If there's any intermediate results to stream.  Stream them.
                if ctx['flask_res']:
                    for res in ctx['flask_res']:
                        yield res
                    ctx['flask_res'] = None

        return Response(stream(), mimetype='text/event-stream')
    else:
        print("calling on_message")
        next_component = self_{self_name}.on_message
        while next_component:
            next_component = next_component.do(ctx)

        print("returning result")
        return ctx_{ctx_name}['flask_res']
        """
        exec(code, globals(), locals())



@xai_component
class ConverseRespond(Component):
    response: InCompArg[str]

    def execute(self, ctx) -> None:

        chat_id = f"chatcmpl-{make_id()}"
        created = int(time.time())

        app = ctx['flask_app']

        with app.app_context():
            ctx['flask_res'] = jsonify(
                {
                    "id": chat_id,
                    "object": "chat.completion",
                    "created": created,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": self.response.value,
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                }
            )




def make_content_response(content, chat_id, created, model_name):
    return json.dumps({
"choices": [
{
"delta": {
"content": content
},
"finish_reason": None,
"index": 0
}
],
"created": created,
"id": chat_id,
"model": model_name,
"object": "chat.completion.chunk"
})


def make_finish_response(model_name, created, chat_id):
    return json.dumps({
"choices": [
{
"delta": {},
"finish_reason": "stop",
"index": 0
}
],
"created": created,
"id": chat_id,
"model": model_name,
"object": "chat.completion.chunk"
})


def stream_answer(ctx, stream, chat_id, created):
    print("called stream_answer")
    for chunk in stream:
        yield f"data: {make_content_response(chunk, chat_id, created, ctx['converse_model_name'])}\n\n"
        time.sleep(0.05)
    yield f"data: {make_finish_response(ctx['converse_model_name'], created, chat_id)}\n\n"


@xai_component
class ConverseStreamRespond(Component):
    response: InCompArg[any]

    def execute(self, ctx) -> None:
        chat_id = f"chatcmpl-{make_id()}"
        created = int(time.time())

        print("making response")
        ctx['flask_res'] = stream_answer(ctx, self.response.value, chat_id, created)


def stream_partial_answer(ctx, stream, chat_id, created):
    print("called stream_answer")
    for chunk in stream:
        yield f"data: {make_content_response(chunk, chat_id, created, ctx['converse_model_name'])}\n\n"

@xai_component
class ConverseStreamPartialResponse(Component):
    response: InCompArg[any]

    def execute(self, ctx) -> None:
        chat_id = f"chatcmpl-{make_id()}"
        created = int(time.time())

        print("making response")
        ctx['flask_res'] = stream_partial_answer(ctx, self.response.value, chat_id, created)


@xai_component
class ConverseRunTool(Component):
    chat_response: InArg[str]

    did_have_tool: OutArg[bool]
    out_response: OutArg[str]

    def execute(self, ctx) -> None:
        text = self.chat_response.value
        print(text)
        self.did_have_tool.value = 'TOOL:' in text

        if self.did_have_tool.value:
            lines = text.split("\n")
            for line in lines:
                if line.startswith('TOOL:'):
                    if "popeye" in line:
                        try:
                            # Run the command and capture the output
                            completed_process = subprocess.run(
                                ['./popeye', '--insecure-skip-tls-verify', '--kubeconfig', 'kubeconfig.yaml', '-o', 'jurassic'],
                                check=True,
                                text=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE
                            )
                            # Set the standard output as the output
                            self.out_response.value = completed_process.stdout
                        except subprocess.CalledProcessError as e:
                            # If there is an error, capture the output and the error message
                            self.out_response.value = e.output if e.output else e.stderr
                        break
                    elif "kubectl" in line:
                        # Extract the command after "TOOL: kubectl"
                        command = line.split("kubectl", 1)[1].strip()

                        try:
                            # Run the command and capture the output
                            completed_process = subprocess.run(
                                ['./kubectl', '--insecure-skip-tls-verify', '--kubeconfig', 'kubeconfig.yaml'] + command.split(),
                                check=True,
                                text=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE
                            )
                            # Set the standard output as the output
                            self.out_response.value = completed_process.stdout
                        except subprocess.CalledProcessError as e:
                            # If there is an error, capture the output and the error message
                            self.out_response.value = e.output if e.output else e.stderr
                        break



@xai_component
class ConverseProcessCommand(Component):
    on_command: BaseComponent

    command_string: InCompArg[str]
    chat_response: InCompArg[str]

    command: OutArg[str]
    did_have_tool: OutArg[bool]
    result_list: OutArg[list]

    def execute(self, ctx) -> None:
        text = self.chat_response.value
        self.did_have_tool.value = self.command_string.value in text
        self.result_list.value = []

        if self.did_have_tool.value:
            lines = text.split("\n")
            for line in lines:
                if line.startswith(self.command_string.value):
                    command = line.split(":", 1)[1].strip()
                    self.command.value = command
                    try:
                        if hasattr(self, 'on_command'):
                            comp = self.on_command
                            while comp is not None:
                                comp = comp.do(ctx)
                    except Exception as e:
                        print(e)
                        