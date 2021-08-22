from selenium import webdriver
from selenium.webdriver.chrome.webdriver import WebDriver
import json
from selenium.webdriver.support.events import (
    EventFiringWebDriver,
    AbstractEventListener,
)


class MyListener(AbstractEventListener):
    def before_click(self, element, driver) -> None:
        print("Clicked on element with text: " + element.text)
        return super().before_click(element, driver)

    def after_click(self, element, driver) -> None:
        print("Clicked on: " + element.text)
        return super().after_click(element, driver)

    def after_navigate_to(self, url, driver) -> None:
        print("Navigated to: " + url)
        return super().after_navigate_to(url, driver)

    def after_navigate_back(self, driver) -> None:
        print("Navigated back")
        return super().after_navigate_back(driver)


with open("queries_and_comment_by_reaction_score_updated.json", "r") as f:
    queries = json.load(f)

my_idx = range(3301 - 1, 3350 - 1 + 1)

my_queries = [queries[i] for i in my_idx]

print(my_queries[0]["issue_question"]["title"])

start_idx = 0

driver: WebDriver = webdriver.Chrome()
driver: EventFiringWebDriver = EventFiringWebDriver(driver, MyListener())

driver.get(my_queries[0]["issue_question"]["url"])
