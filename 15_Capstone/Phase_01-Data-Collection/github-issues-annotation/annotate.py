import sys
from typing import Tuple

from PyQt6.QtCore import QModelIndex

# sys.path.append("qt-material")

from PyQt6 import QtGui, QtWidgets
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QHBoxLayout,
    QLabel,
    QListView,
    QProgressBar,
    QPushButton,
    QWidget,
    QVBoxLayout,
)

# from qt_material import apply_stylesheet
from bs4 import BeautifulSoup
import json
from pathlib import Path
import requests
from graphql_utils import run_query, query, headers

# from qt_material import apply_stylesheet
from dataclasses import dataclass, field


@dataclass
class QueryDetails:
    query_question: str
    query_url: str
    issue_id: int
    repo_user: str
    repo_name: str


def dict_raise_on_duplicates(ordered_pairs):
    """Reject duplicate keys."""
    d = {}
    for k, v in ordered_pairs:
        if k in d:
            #    raise ValueError("duplicate key: %r" % (k,))
            pass
        else:
            d[k] = v
    return d


@dataclass
class AnnotationsSaver:
    save_file: str

    def __init__(self, file_name) -> None:
        self.save_file = file_name

        if Path(self.save_file).exists():
            with open(self.save_file, "r") as f:
                self.annotations = json.load(
                    f, object_pairs_hook=dict_raise_on_duplicates
                )
        else:
            self.annotations = {}

    def add_annotation(self, issue_id, question, answer):
        self.annotations[str(issue_id)] = {"question": question, "answer": answer}

    def save(self):
        # print(self.annotations)
        with open(self.save_file, "w") as f:
            json.dump(self.annotations, f)


@dataclass
class AnnotationsModel:
    current_idx: int = 0
    queries: list = field(default_factory=list)
    my_idx: list = field(default_factory=list)
    my_queries: list = field(default_factory=list)

    def __init__(
        self,
        issues_range: Tuple[int, int] = [3301 - 1, 3350 - 1 + 1],
        start_idx: int = 0,
    ) -> None:
        self.current_idx = start_idx

        with open("queries_and_comment_by_reaction_score_updated.json", "r") as f:
            self.queries = json.load(f)

        self.my_idx = list(range(*issues_range))

        self.my_queries = [self.queries[i] for i in self.my_idx]
        print(f"Total Queries: {len(self.my_queries)}")

    def get_query_details(self) -> QueryDetails:
        query_question = self.my_queries[self.current_idx]["issue_question"]["title"]
        query_url = self.my_queries[self.current_idx]["issue_question"]["url"]
        issue_id = int(query_url.split("/")[-1])
        repo_name, repo_user = query_url.split("/")[-4:-2]

        return QueryDetails(query_question, query_url, issue_id, repo_user, repo_name)

    def set_next_query(self) -> None:
        self.current_idx += 1

        self.current_idx = min(self.current_idx, len(self.my_queries) - 1)

    def set_prev_query(self) -> None:
        self.current_idx -= 1

        self.current_idx = max(0, self.current_idx)

    def __len__(self) -> int:
        return len(self.my_queries)

    def get_query_comments(self) -> list:

        query_details = self.get_query_details()

        print(f"Running Query for Issue ID: {query_details.issue_id}")

        gql_variables = {
            "repoName": query_details.repo_name,
            "repoOwner": query_details.repo_user,
            "issueNumber": query_details.issue_id,
        }

        result = run_query(query=query, variables=gql_variables, headers=headers)
        result = result["data"]["repository"]["issue"]["comments"]["nodes"]
        result = [x["body"] for x in result]

        # print(result)
        return result


class Widget(QtWidgets.QWidget):
    FONT = QtGui.QFont("Arial", 24)

    def __init__(self, model: AnnotationsModel):
        super().__init__()
        self.model: AnnotationsModel = model
        self.saver: AnnotationsSaver = AnnotationsSaver("annotations.json")
        self.init_font()
        self.init_ui()

    def init_font(self):
        font_id = QtGui.QFontDatabase.addApplicationFont(
            "fonts/static/JetBrainsMono-Regular.ttf"
        )
        _fontstr = QtGui.QFontDatabase.applicationFontFamilies(font_id)[0]
        self.TITLE_FONT = QtGui.QFont(_fontstr, 24)
        self.TITLE_FONT.setBold(True)
        self.FONT = QtGui.QFont(_fontstr, 12)

    def init_ui(self):
        self.setWindowTitle("Annotate")
        self.setGeometry(300, 300, 600, 600)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        title = QLabel("Github Comments Annotation Tool")
        title.setFont(self.TITLE_FONT)
        layout.addWidget(title)

        query_data = self.model.get_query_details()
        self.question_title = QLabel(f"Question: {query_data.query_question}")
        self.question_title.setFont(self.FONT)
        layout.addWidget(self.question_title)

        self.listView: QListView = QListView()
        self.listView.setFont(self.FONT)
        self.listView.setStyleSheet(
            """
        QListView::item::!selected {
            margin-top: 15px;
            border-bottom: 1px solid black;
        }
        """
        )

        layout.addWidget(self.listView)

        self.nav_layout = QHBoxLayout()

        self.next = QPushButton("Next >")
        self.next.setFont(self.FONT)
        self.next.clicked.connect(self.next_query)
        self.shortcut_next = QtGui.QShortcut(QtGui.QKeySequence("Right"), self)
        self.shortcut_next.activated.connect(self.next_query)

        self.back = QPushButton("< Back")
        self.back.setFont(self.FONT)
        self.back.clicked.connect(self.prev_query)
        self.shortcut_back = QtGui.QShortcut(QtGui.QKeySequence("Left"), self)
        self.shortcut_back.activated.connect(self.prev_query)

        self.nav_layout.addWidget(self.back)
        self.nav_layout.addWidget(self.next)

        layout.addLayout(self.nav_layout)

        self.progress_layout = QHBoxLayout()

        self.status = QLabel(f"Progress {self.model.current_idx}/{len(self.model)}")
        self.status.setFont(self.FONT)
        self.progress_layout.addWidget(self.status)

        self.prog_bar = QProgressBar()
        self.prog_bar.setFont(self.FONT)
        self.prog_bar.setValue((self.model.current_idx / len(self.model)) * 100)
        self.progress_layout.addWidget(self.prog_bar)

        layout.addLayout(self.progress_layout)

        self.data_entry = QtGui.QStandardItemModel()
        self.listView.setModel(self.data_entry)
        self.listView.clicked.connect(self.on_clicked)

        comments = self.model.get_query_comments()

        for text in comments:
            item = QtGui.QStandardItem(text)
            self.data_entry.appendRow(item)

        self.listView.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)

        self.show()

    def next_query(self):
        self.model.set_next_query()
        self.prog_bar.setValue((self.model.current_idx / len(self.model)) * 100)
        self.status.setText(f"Progress {self.model.current_idx}/{len(self.model)}")
        self.data_entry.clear()
        comments = self.model.get_query_comments()
        question = self.model.get_query_details().query_question
        self.question_title.setText(f"Question: {question}")

        for text in comments:
            item = QtGui.QStandardItem(text)
            self.data_entry.appendRow(item)

    def prev_query(self):
        self.model.set_prev_query()
        self.prog_bar.setValue((self.model.current_idx / len(self.model)) * 100)
        self.status.setText(f"Progress {self.model.current_idx}/{len(self.model)}")
        self.data_entry.clear()
        comments = self.model.get_query_comments()
        question = self.model.get_query_details().query_question
        self.question_title.setText(f"Question: {question}")

        for text in comments:
            item = QtGui.QStandardItem(text)
            self.data_entry.appendRow(item)

    def on_clicked(self, index: QModelIndex):
        # self.data_entry.clear()
        print(f"Clicked on {index.data()}")

        self.saver.add_annotation(
            self.model.get_query_details().issue_id,
            self.model.get_query_details().query_question,
            index.data(),
        )

        # save this annotation
        self.saver.save()

        self.next_query()


start_idx = 0


def main():
    app = QApplication([])

    annot_model = AnnotationsModel(start_idx=194, issues_range=(3301 - 1, 3850 - 1 + 1))

    w = Widget(model=annot_model)

    # apply_stylesheet(app, theme="dark_teal.xml", extra={"font-family": "Roboto"})

    # w.show()
    w.showMaximized()

    app.exec()


if __name__ == "__main__":
    main()
