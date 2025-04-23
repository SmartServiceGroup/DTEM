import requests


GITHUB_TOKEN = ''

# 替换为目标用户的 node_id
NODE_ID = ''

# GraphQL 查询
query = """
query($node_id: ID!) {
  node(id: $node_id) {
    ... on User {
      repositoriesContributedTo(first: 100, contributionTypes: [COMMIT], includeUserRepositories: true) {
        nodes {
          name
          owner {
            login
          }
          url
        }
      }
    }
  }
}
"""

# 设置请求头
headers = {
    'Authorization': f'Bearer {GITHUB_TOKEN}',
    'Content-Type': 'application/json'
}

# 设置变量
variables = {'node_id': NODE_ID}

# 发送请求
response = requests.post(
    'https://api.github.com/graphql',
    json={'query': query, 'variables': variables},
    headers=headers
)

# 解析响应
if response.status_code == 200:
    data = response.json()
    repositories = data['data']['node']['repositoriesContributedTo']['nodes']
    for repo in repositories:
        print(f"Repository: {repo['owner']['login']}/{repo['name']}")
        print(f"URL: {repo['url']}\n")
else:
    print(f"Query failed to run by returning code of {response.status_code}.")
    print(response.text)
