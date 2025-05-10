
import ipywidgets as widgets
from IPython.display import display
import json

class Log:
    
    def __init__(self, log_id = -1):
        self.log_dict_ids = []
        self.selected_id = 0
        self.all_log_dicts = {}
        self.mode = 0

        # for menu
        self.menu_elements = ["Default display mode", "Display all experts categorized by class"]
        self.menu_selected_id = 0
        self.is_menu = False
        
        # for mode 0
        with open("log.json") as json_file:
            self.logs = json.load(json_file)
        all_log_dict = self.logs[log_id]
        self.all_log_dicts[0] = all_log_dict

        # for mode 1
        all_log_dict2 = self.make_log_dict2()
        self.all_log_dicts[1] = all_log_dict2

        # start with default display mode
        self.log_dict = self.all_log_dicts[0]


    
    def get_log_dict_text(self):

        # <span style='color:gray;'>
        text = "\n<pre>" + self.log_dict["expert_class_name"] + "\n"

        for i in range(len(self.log_dict["called_experts"])):
            if i == self.selected_id:
                text += "<span style='color:green;'>    " + self.log_dict["called_experts"][i]["expert_class_name"] + "</span>\n"
                for j in range(len(self.log_dict["called_experts"][i]["called_experts"])):
                    text += "       " + self.log_dict["called_experts"][i]["called_experts"][j]["expert_class_name"] + "\n"
            else:
                text += "    " + self.log_dict["called_experts"][i]["expert_class_name"] + "\n"
            
        text += "</pre>"

        text += self.get_arg_return_text()
    
        return text


    def get_menu_text(self):
        elements = self.menu_elements
        selected_id = self.menu_selected_id

        text = "<pre>"
    
        for i in range(len(self.menu_elements)):
            if i == self.menu_selected_id:
                text += "<span style='color:green;'>" + self.menu_elements[i] + "</span>\n"
            else:
                text += self.menu_elements[i] + "\n"
            
        text += "</pre>"

        return text


    # for showing called_experts
    def make_log_dict2(self):

        experts = {}

        # recursive function
        def process_expert_dict(log_dict_ids):
            expert_dict = self.get_log_dict_from_log_dict_ids(log_dict_ids)
            expert_class_name = expert_dict["expert_class_name"]
            expert_dict["log_dict_ids"] = log_dict_ids
            if expert_class_name not in experts:
                experts[expert_class_name] = [expert_dict]
            else:
                experts[expert_class_name].append(expert_dict)

            for new_id in range(len(expert_dict["called_experts"])):
                process_expert_dict(log_dict_ids + [new_id])

        process_expert_dict([])

        log_dict2 = {"expert_class_name":"Expert", "args":None, "return":None, "called_experts":[]}
        for key in experts:
            output_dict_ = {"expert_class_name":key, "args":None, "return":None, "called_experts":experts[key]}
            log_dict2["called_experts"].append(output_dict_)

        return log_dict2  # {"expert_class_name":"Expert", "called_experts":[{"expert_class_name": "expert1", "called_experts":[log_dict1, ...]}, ]}
        

    def get_log_dict_from_log_dict_ids(self, log_dict_ids):
        log_dict = self.all_log_dicts[self.mode]
        for id in log_dict_ids:
            log_dict = log_dict["called_experts"][id]
        return log_dict

    
    def get_arg_return_text(self):
        text = f"""<pre>\n\n--- args ---\n{self.json_show(self.log_dict["called_experts"][self.selected_id]["args"], 0)}\n\n"""
        text += f"""--- return ---\n{self.json_show(self.log_dict["called_experts"][self.selected_id]["return"], 0)}</pre>"""
        text = text.replace("<s>","")
        return text
        

    def json_show(self, element, num_column):
        text = ""

        if isinstance(element, list):
            text += " " * 3 * num_column + "[\n"
            for i, e in enumerate(element):
                text += " " * 3 * (num_column + 1) + f"- {i+1} -\n"
                text += self.json_show(e, num_column + 1) + "\n"
            text += " " * 3 * num_column + "]\n"

        elif isinstance(element, dict):
            for i, key in enumerate(element):
                text += " " * 3 * num_column + f"- {i+1} - " + key + " :\n"
                text += self.json_show(element[key], num_column + 1) + "\n"

        elif isinstance(element, (str, int, bool)) or element is None:
            text += " " * 3 * num_column + str(element) + "\n"

        else:
            raise Exception("element must be list, dict, str, int, or bool")

        return text
        
        
    def show(self):

        text_display = widgets.HTML(value=self.get_log_dict_text())
        
        # Define functions to handle button clicks
        def on_up_button_clicked(b):
            if self.is_menu:
                if self.menu_selected_id > 0:
                    self.menu_selected_id -= 1
                text_display.value = self.get_menu_text()
            else:
                if self.selected_id > 0:
                    self.selected_id -= 1
                text_display.value = self.get_log_dict_text()
        
        def on_down_button_clicked(b):
            if self.is_menu:
                if self.menu_selected_id < len(self.menu_elements) - 1:
                    self.menu_selected_id += 1
                text_display.value = self.get_menu_text()
            else:
                if self.selected_id < len(self.log_dict["called_experts"]) - 1:
                    self.selected_id += 1
                text_display.value = self.get_log_dict_text()
        
        def on_left_button_clicked(b):
            if self.is_menu:
                pass
            else:
                if self.log_dict_ids!=[]: self.selected_id = self.log_dict_ids.pop()
                self.log_dict = self.all_log_dicts[self.mode]
                for id in self.log_dict_ids:
                    self.log_dict = self.log_dict["called_experts"][id]
                text_display.value = self.get_log_dict_text()
        
        def on_right_button_clicked(b):
            if self.is_menu:
                pass
            else:
                if self.log_dict["called_experts"] != []:
                    self.log_dict = self.log_dict["called_experts"][self.selected_id]
                    self.log_dict_ids.append(self.selected_id)
                    self.selected_id = 0
                text_display.value = self.get_log_dict_text()
        
        def on_center_button_clicked(b):
            if self.is_menu:
                self.mode = self.menu_selected_id
                self.is_menu = False
                if self.mode == 0:  # when going to mode 0 from another mode, the expert being selected will be displayed at first
                    log_dict_ids = self.log_dict["called_experts"][self.selected_id]["log_dict_ids"]
                    self.selected_id = log_dict_ids[-1]
                    self.log_dict_ids = log_dict_ids[:-1]
                    self.log_dict = self.get_log_dict_from_log_dict_ids(self.log_dict_ids)
                else:
                    self.log_dict_ids = []
                    self.log_dict = self.all_log_dicts[self.mode]
                text_display.value = self.get_log_dict_text()
            else:
                text = self.get_log_dict_text()
                #text += self.get_arg_return_text()
                text_display.value = text
        
        def on_left_up_button_clicked(b):
            self.is_menu = True
            text = self.get_menu_text()
            text_display.value = text

        up_button = widgets.Button(description='Up')
        down_button = widgets.Button(description='Down')
        left_button = widgets.Button(description='Back')
        right_button = widgets.Button(description='Next')
        center_button = widgets.Button(description='Select')
        left_up_button = widgets.Button(description='Menu')

        up_button.on_click(on_up_button_clicked)
        down_button.on_click(on_down_button_clicked)
        left_button.on_click(on_left_button_clicked)
        right_button.on_click(on_right_button_clicked)
        center_button.on_click(on_center_button_clicked)
        left_up_button.on_click(on_left_up_button_clicked)

        buttons = [
            left_up_button,
            up_button,
            widgets.Button(description=''),
            left_button,
            center_button,
            right_button,
            widgets.Button(description=''),
            down_button,
            widgets.Button(description=''),
        ]

        grid = widgets.GridBox(children=buttons,
                               layout=widgets.Layout(grid_template_columns='repeat(3, 150px)',
                                                     grid_template_rows='repeat(3, 30px)',
                                                     grid_gap='10px'))

        # Create a text input widget
        text_input = widgets.Text(
            value='',
            placeholder='W:Up, A:Left, Z:Down, D:Right, S:Select, Q:Menu',
            #description='Input:',
            disabled=False
        )

        # Define a function to handle the input
        def handle_input(change):
            #lobal text_input
            #with output:
            text_input.value = ''
            user_input = change['new']
            if user_input == 'w':
                on_up_button_clicked(None)
            elif user_input == 'a':
                on_left_button_clicked(None)
            elif user_input == 'z' or user_input == 'x':
                on_down_button_clicked(None)
            elif user_input == 'd':
                on_right_button_clicked(None)
            elif user_input == 's':
                on_center_button_clicked(None)
            elif user_input == 'q':
                on_left_up_button_clicked(None)

        # Observe changes in the text input widget
        text_input.observe(handle_input, names='value')

        display(grid, text_input, text_display)
