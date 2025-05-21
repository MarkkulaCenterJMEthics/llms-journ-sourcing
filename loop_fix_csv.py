import glob
import os
import json
import csv

root_folder = "/home/shawnyang/Documents/Project/llms-journ-sourcing/llm_results/llms_20250504210825/gemini-2.5-flash/"

for folder in glob.glob(f"{root_folder}/*"):
    for json_file in glob.glob(f"{folder}/*.json"):
        csv_file = json_file.replace(".json", ".csv")
        if True: 
            # print(json_file)

            try:
                with open(json_file, "r") as f:
                    json_data = json.load(f)

                data = []

                for source_type in [
                    "Named Organization Sources",
                    "Named Person Sources",
                    "Document Sources",
                    "Unnamed Group of People",
                    "Anonymous Sources",
                ]:
                    for source in json_data.get(source_type, []):
                        name = source.get("Name of Source", "")
                        title = source.get("Title of Source", "")
                        association = source.get("Source Justification", "")
                        sourced_statements = source.get("Sourced Statement", [])

                        if isinstance(sourced_statements, list):
                            for sourced_statement in sourced_statements:
                                data.append(
                                    [
                                        sourced_statement,
                                        source_type,
                                        name,
                                        title,
                                        association,
                                    ]
                                )
                        else:
                            data.append(
                                [
                                    sourced_statements,
                                    source_type,
                                    name,
                                    title,
                                    association,
                                ]
                            )

                with open(csv_file, "w", newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(
                        [
                            "SourcedStatement",
                            "SourceType",
                            "Name",
                            "Title",
                            "Justification",
                        ]
                    )
                    writer.writerows(data)

                # print(f"CSV file generated: {csv_file}")
            except Exception as e:
                print(json_file)
                # print(f"CSV not generated for {csv_file}: {e}")
