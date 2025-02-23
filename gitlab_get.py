import os
import gitlab
import base64
import json
from dotenv import load_dotenv

load_dotenv('.env')
GITLAB_URL = os.environ.get('GITLAB_URL')
GITLAB_KEY = os.environ.get('GITLAB_KEY')

datalist= []

# private token authentication
gl = gitlab.Gitlab(GITLAB_URL, private_token=GITLAB_KEY)
gl.auth()

fileinfo = {'README.md': 'Show README file from project',
            'requirements.txt': 'List the project dependencies',
            'config.properties': 'Show configurations from project',
            'Dockerfile': 'Show Dockerfile from project'}

# list all projects
projects = gl.projects.list(all=True)
for project in projects:

    pj_name =  project.name
    print(f'=============> Projeto {pj_name}')
    # Skip projects without branches
    if len(project.branches.list()) == 0:
        continue

    branch = project.branches.list()[0].name

    for fname in fileinfo:

        try:
            f = project.files.get(file_path=fname, ref=branch)
        except gitlab.exceptions.GitlabGetError:
            # Skip projects without Dockerfile
            continue

        file_content = base64.b64decode(f.content).decode("utf-8")
        file_content = file_content.replace('\\n', '\n')

        print(f'{fname}: {file_content}')
        print('-' * 40)

        dc = {'conversations': []}
        dc['conversations'].append(
                {"from": "system", "value": "Assistente projetos GPr"}
        )
        dc['conversations'].append(
                {"from": "human", "value": f"{fileinfo[fname]} {pj_name}"}
        )
        dc['conversations'].append(
                {"from": "gpt", "value": f"{file_content}"}
        )

        datalist.append(dc)


# Save data do file
with open("data.jsonl", "w", encoding="ascii") as f:
    for sample in datalist:
        json_line = json.dumps(sample, ensure_ascii=False)
        try:
            f.write(json_line + "\n")
        except:
            pass
