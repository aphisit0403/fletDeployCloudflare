from PerfectcomAPI import OcrReader
import flet as ft
import json


test = OcrReader()

def main(page: ft.Page):
    page.scroll = "auto"

    image_loc = ft.TextField(label="your image name")
    Identification_Number = ft.TextField(label="Identification_Number")
    PrefixTH = ft.TextField(label="PrefixTH")
    NameTH = ft.TextField(label="NameTH")
    LastNameTH = ft.TextField(label="LastNameTH")
    PrefixEN = ft.TextField(label="PrefixEN")
    NameEN = ft.TextField(label="NameEN")
    LastNameEN = ft.TextField(label="LastNameEN")
    BirthdayTH = ft.TextField(label="BirthdayTH")
    BirthdayEN = ft.TextField(label="BirthdayEN")
    Religion = ft.TextField(label="Religion")
    Address = ft.TextField(label="Address")
    Issuedate_TH = ft.TextField(label="Issuedate_TH")
    Expirydate_TH = ft.TextField(label="Expirydate_TH")
    Issuedate_EN = ft.TextField(label="Issuedate_EN")
    Expirydate_EN = ft.TextField(label="Expirydate_EN")

    def processyouimage(e):
        text = test.extract_data(image_loc.value)
        card_info = []
        for i in text:
            card_info.append(i)
        try:
            dictionary = {
                "Identification_Number": card_info[0],
                "PrefixTH": card_info[1],
                "NameTH": card_info[2],
                "LastNameTH": card_info[3],
                "PrefixEN": card_info[4],
                "NameEN": card_info[5],
                "LastNameEN": card_info[6],
                "BirthdayTH": card_info[7],
                "BirthdayEN": card_info[8],
                "Religion": card_info[9],
                "Address": card_info[10],
                "Issuedate_TH":card_info[11],
                "Expirydate_TH":card_info[13],
                "Issuedate_EN":card_info[12],
                "Expirydate_EN":card_info[14]
            }

            json_object = json.dumps(dictionary, indent=14, ensure_ascii=False)

            # Writing to sample.json
            with open("sample.json", "w", encoding="utf-8") as outfile:
                outfile.write(json_object)

            json_dict =json.loads(json_object)
            Identification_Number.value = json_dict["Identification_Number"]
            PrefixTH.value = json_dict["PrefixTH"]
            NameTH.value = json_dict["NameTH"]
            LastNameTH.value = json_dict["LastNameTH"]
            PrefixEN.value = json_dict["PrefixEN"]
            NameEN.value = json_dict["NameEN"]
            LastNameEN.value = json_dict["LastNameEN"]
            BirthdayTH.value = json_dict["BirthdayTH"]
            BirthdayEN.value = json_dict["BirthdayEN"]
            Religion.value = json_dict["Religion"]
            Address.value = json_dict["Address"]
            Issuedate_TH.value = json_dict["Issuedate_TH"]
            Expirydate_TH.value = json_dict["Expirydate_TH"]
            Issuedate_EN.value = json_dict["Issuedate_EN"]
            Expirydate_EN.value = json_dict["Expirydate_EN"]
        except:
            pass

        page.snack_bar = ft.SnackBar(
            ft.Text("success get from image",size=30),
            bgcolor="green"
            )
        page.snack_bar.open = True
        page.update()


    page.add(
        ft.Column([
            image_loc,
            ft.ElevatedButton("Process you image",
                           bgcolor="blue", color="white",
                           on_click=processyouimage
                           ),
            ft.Text("You Result in image", weight="bold"),
            Identification_Number,
            PrefixTH,
            NameTH,
            LastNameTH,
            PrefixEN,
            NameEN,
            LastNameEN,
            BirthdayTH,
            BirthdayEN,
            Religion,
            Address,
            Issuedate_TH,
            Expirydate_TH,
            Issuedate_EN,
            Expirydate_EN
        ])
    )


ft.app(target=main)

