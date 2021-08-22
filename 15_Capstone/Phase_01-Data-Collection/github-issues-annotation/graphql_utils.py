import requests
from API_KEYS import GITHUB_TOKEN

headers = {"Authorization": f"Bearer {GITHUB_TOKEN}" }

def run_query(query, variables, headers): # A simple function to use requests.post to make the API call. Note the json= section.
    request = requests.post('https://api.github.com/graphql', json={'query': query, 'variables': variables}, headers=headers)
    if request.status_code == 200:
        return request.json()
    else:
        raise Exception("Query failed to run by returning code of {}. {}".format(request.status_code, query))

query = """
query GetComments($repoName: String!, $repoOwner: String!, $issueNumber: Int!) {
  repository(name: $repoName, owner: $repoOwner) {
    issue(number: $issueNumber) {
      comments(first: 100) {
        nodes {
          body
        }
      }
    }
  }
}
"""