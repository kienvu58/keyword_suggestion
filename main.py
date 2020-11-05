import pandas as pd
import numpy as np
import time
import preprocess
from apyori import apriori


class Miner:
    def __init__(self):
        pass

    def get_data(
        self,
        dsn_database,
        dsn_hostname,
        dsn_port,
        dsn_protocol,
        dsn_uid,
        dsn_pwd,
        level,
    ):
        preprocess = preprocess.Preprocessor()
        raw_data = preprocess.db2_connect(
            dsn_database, dsn_hostname, dsn_port, dsn_protocol, dsn_uid, dsn_pwd
        )
        data = preprocess.data_preprocess(raw_data, level)
        return data

    def get_itemset(self, data):
        print("Start generating itemsets...")
        start = time.time()
        list_itemset = []
        for i in range(data.shape[0]):
            itemset = data.iloc[i].dropna().index.tolist()[1:]
            list_itemset.append(itemset)
            list_itemset = np.array(list_itemset)
        print(
            "Done generating itemsets. Time taken = {:.1f}(s) \n".format(
                time.time() - start
            )
        )
        return list_itemset

    def assoc_rules(
        self,
        list_itemset,
        min_support=0.03,
        min_confidence=0.10,
        min_lift=2,
        min_length=2,
    ):
        print("Start mining frequent itemsets...")
        start = time.time()
        association_rules = apriori(
            list_itemset, min_support, min_confidence, min_lift, min_length
        )
        association_results = np.array(association_rules)
        print(
            "Done mining frequent itemsets. Time taken = {:.1f}(s) \n".format(
                time.time() - start
            )
        )
        return association_results

    def item_suggest(self, association_results, mapname, name=True):
        print("Start generating frequent itemsets...")
        start = time.time()
        results = []
        for item in association_results:
            pair = item[0]
            items = [x for x in pair]
            value0 = str(items[0])
            value1 = str(items[1])
            value2 = str(item[1])[:7]
            value3 = str(item[2][0][2])[:7]
            value4 = str(item[2][0][3])[:7]
            rows = (value0, value1, value2, value3, value4)
            results.append(rows)
        labels = ["Interest_ID 1", "Interest_ID 2", "Support", "Confidence", "Lift"]
        item_suggestion = pd.DataFrame.from_records(results, columns=labels)
        if name:
            interest_flat = pd.read_csv(mapname + ".csv", header=0, dtype=object)
            item_labels = dict(zip(interest_flat["ID"], interest_flat["NAME"]))
            item_suggestion["Interest_name 1"] = item_suggestion["Interest_ID 1"].map(
                item_labels
            )
            item_suggestion["Interest_name 2"] = item_suggestion["Interest_ID 2"].map(
                item_labels
            )
            item_suggestion = item_suggestion[
                [
                    "Interest_ID 1",
                    "Interest_name 1",
                    "Interest_ID 2",
                    "Interest_name 2",
                    "Support",
                    "Confidence",
                    "Lift",
                ]
            ]
        else:
            item_suggestion = item_suggestion
        print(
            "Done generating frequent itemsets. Time taken = {:.1f}(s) \n".format(
                time.time() - start
            )
        )
        return item_suggestion


if __name__ == "__main__":
    dsn_driver = "IBM DB2 ODBC DRIVER"
    dsn_database = "XXX"
    dsn_hostname = "XXX"
    dsn_port = "XXX"
    dsn_protocol = "XXX"
    dsn_uid = "XXX"
    dsn_pwd = "XXX"
    level = "XXX"
    mapname = "XXX"
    miner = Miner()
    data = miner.get_data(
        dsn_database, dsn_hostname, dsn_port, dsn_protocol, dsn_uid, dsn_pwd, level
    )
    list_itemset = miner.get_itemset(data)
    association_results = miner.assoc_rules(
        list_itemset, min_support=0.03, min_confidence=0.10, min_lift=2, min_length=2
    )
    item_suggestion = miner.item_suggest(association_results, mapname, name=True)
