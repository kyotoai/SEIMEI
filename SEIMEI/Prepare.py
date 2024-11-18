import os, json, time, random, torch
from transformers import AutoTokenizer
from flask import Flask, render_template, request, jsonify


class Prepare:
    def __init__(self, database_path, save_path, rules, file_info, model_name = "gpt2", max_tokens = 10000, min_tokens = 3000):
        self.database_path = database_path
        self.save_path = save_path
        self.rules = rules
        self.file_info = file_info
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, 
            padding_side="left",
            add_eos_token=False,
            add_bos_token=False,)

        if not os.path.exists(database_path):
            os.makedirs(database_path)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.file_paths = []
        for i in range(len(file_info)):
            full_folder_path = os.path.join(database_path, file_info[i]["folder_path"])
            filenames_in_a_directory = os.listdir(full_folder_path)
            for file_name in filenames_in_a_directory:
                for extension in file_info[i]["extensions"]:
                    if file_name.endswith(extension):
                        path = os.path.join(full_folder_path, file_name)
                        if not os.path.isdir(path):
                            self.file_paths.append(path)


    def make_chunks(self):

        def split_into_chunks(tokenizer, text, max_tokens, min_tokens, rules, debug):
            chunk_list = []
            instruction_list = []
            warnings = []
            split_rule_keys = []
            
            start_id = 0
            tokenized_text = tokenizer(text, return_tensors="pt", add_special_tokens = False)
            num_tokens = len(tokenized_text["input_ids"][0])
            #text_size = len(text)
            rest_text = text
            
            for i in range(int(num_tokens/min_tokens)+1):
                    
                tokenized_text = tokenizer(rest_text, return_tensors="pt", add_special_tokens = False)
                
                if len(tokenized_text["input_ids"][0]) < max_tokens:
                    process_tokenized_text = tokenized_text["input_ids"][0]
                    processed_text = tokenizer.decode(process_tokenized_text, skip_special_tokens=True)
                    chunk_list.append(processed_text)
                    instruction_list.append(processed_text)
                    warnings.append(0)
                    split_rule_keys.append("[END]")
                    break

                process_tokenized_text = tokenized_text["input_ids"][0][:max_tokens]
                process_text1 = tokenizer.decode(process_tokenized_text, skip_special_tokens=True) # process_text1 is only for getting approximate max_tokens of text
                process_text_size = len(process_text1)
                process_text = rest_text[:process_text_size]

                #determine where should be split
                min_split_text = process_text_size
                is_text_split = False
                #warning = 1
                for j in range(len(rules)):
                    for rule_key in rules[j].keys():
                        split_process_text = process_text.split(rule_key)
                        if len(split_process_text) > 1:
                            size_last_split_process_text = len(split_process_text[-1])
                            if (size_last_split_process_text < min_split_text) and (size_last_split_process_text < (1 - min_tokens/max_tokens)*process_text_size) and (size_last_split_process_text!=process_text_size):
                                is_text_split = True
                                split_rule_key = rule_key
                                min_split_text = size_last_split_process_text + len(rule_key) * rules[j][rule_key]

                                if debug:
                                    print("=====")
                                    print("rule_key", rule_key)
                                    print()
                                    print("j: ", j)
                                    print()
                                    print("process_text[:-min_split_text]: ", process_text[:-min_split_text])
                                    print()
                                    print("process_text[-min_split_text:]: ", process_text[-min_split_text:])
                                    print()

                                
                                break # this is supposed to be unnecessary, but I saw some weird thing without this for some reason. This cause must be figured out at some point
                                
                    if is_text_split:
                        #if j < warning_id: warning = 0
                        break
                                
                if is_text_split:
                    processed_text = process_text[:-min_split_text]
                    rest_text = rest_text[len(processed_text):]
                    split_rule_keys.append(rule_key)
                else:
                    processed_text = process_text
                    rest_text = rest_text[len(processed_text):]
                    split_rule_keys.append("")
                
                #processed_tokenized_text = tokenizer(processed_text, return_tensors="pt", add_special_tokens = False)
                #len_processed_text = len(processed_tokenized_text["input_ids"][0])  #this could be more than max_tokens without min sentence, which caused fatal error

                if len(processed_text)==0:
                    break

                chunk_list.append(processed_text)
                #instruction_list.append(processed_text)
                #warnings.append(warning)
                
                #start_id += len_processed_text # taking from process_tokenized_text to prevent the id from getting wrong

            return chunk_list  #, instruction_list, warnings, split_rule_keys
        

        start = time.time()

        all_chunks = []
        all_file_paths = []
        chunk_dict = {}
        #split_rule_dict = {}
        #warning_chunk_dict = {}

        for file_path in self.file_paths:
            try:
                with open(file_path) as f:
                    text = f.read()
            except:
                continue
            debug = False
            #if file_path == "../gkv-code/src/gkvp_geom.f90":
            #    debug = True
                
            chunks = split_into_chunks(self.tokenizer, text, self.max_tokens, self.min_tokens, self.rules, debug)  #insts, warnings, split_rule_keys
            
            all_chunks += chunks
            fp = [file_path for i in range(len(chunks))]
            all_file_paths += fp
            
            for i in range(len(chunks)):
                chunk_dict[file_path] = chunks
                #if file_path in chunk_dict: chunk_dict[file_path][file_path+str(i)] = chunks[i]
                #else: chunk_dict[file_path] = {file_path+str(i):chunks[i]}
                
                #if warnings[i] == 1: warning_chunk_dict[file_path+str(i)] = chunks[i]

                #split_rule_dict[file_path] = split_rule_keys
            
        end = time.time()

        print("total num chunk: ", len(all_chunks))
        print("process time: ", end - start)


        input_file_path = f"{self.save_path}/chunks.json"
        with open(input_file_path, 'w') as json_file:
            json.dump(all_chunks, json_file)

        file_path_json = f"{self.save_path}/file_paths.json"
        with open(file_path_json, 'w') as json_file:
            json.dump(all_file_paths, json_file)

        file_path_json = f"{self.save_path}/chunk_dict.json"
        with open(file_path_json, 'w') as json_file:
            json.dump(chunk_dict, json_file)
            
        print("file saved")

        # chunks: [<str> chunk of the text, ...]
        # insts: [<str> instructions corresponds to chunk, ...]




    def make_chunks2(self):

        def split_into_chunks(tokenizer, text, max_tokens, min_tokens, rules, debug):
            chunk_list = []
            instruction_list = []
            warnings = []
            split_rule_keys = []
            
            start_id = 0
            tokenized_text = tokenizer(text, return_tensors="pt", add_special_tokens = False)
            num_tokens = len(tokenized_text["input_ids"][0])
            #text_size = len(text)
            
            for i in range(int(num_tokens/min_tokens)+1):
                
                if(start_id + max_tokens >= num_tokens):
                    process_tokenized_text = tokenized_text["input_ids"][0][start_id:]
                    processed_text = tokenizer.decode(process_tokenized_text, skip_special_tokens=True)
                    chunk_list.append(processed_text)
                    instruction_list.append(processed_text)
                    warnings.append(0)
                    split_rule_keys.append("[END]")
                    break
                    
                process_tokenized_text = tokenized_text["input_ids"][0][start_id : start_id + max_tokens]
                process_text = tokenizer.decode(process_tokenized_text, skip_special_tokens=True) # this should be decoded since subword token is difficult to handle
                process_text_size = len(process_text)

                #determine where should be split
                min_split_text = process_text_size
                is_text_split = False
                #warning = 1
                for j in range(len(rules)):
                    for rule_key in rules[j].keys():
                        split_process_text = process_text.split(rule_key)
                        if len(split_process_text) > 1:
                            size_last_split_process_text = len(split_process_text[-1])
                            if (size_last_split_process_text < min_split_text) and (size_last_split_process_text < (1 - min_tokens/max_tokens)*process_text_size) and (size_last_split_process_text!=process_text_size):
                                is_text_split = True
                                split_rule_key = rule_key
                                min_split_text = size_last_split_process_text + len(rule_key) * rules[j][rule_key]

                                if debug:
                                    print("=====")
                                    print("rule_key", rule_key)
                                    print()
                                    print("j: ", j)
                                    print()
                                    print("process_text[:-min_split_text]: ", process_text[:-min_split_text])
                                    print()
                                    print("process_text[-min_split_text:]: ", process_text[-min_split_text:])
                                    print()

                                
                                break # this is supposed to be unnecessary, but I saw some weird thing without this for some reason. This cause must be figured out at some point
                                
                    if is_text_split:
                        #if j < warning_id: warning = 0
                        break
                                
                if is_text_split:
                    processed_text = process_text[:-min_split_text]
                    split_rule_keys.append(rule_key)
                else:
                    processed_text = process_text
                    split_rule_keys.append("")
                
                processed_tokenized_text = tokenizer(processed_text, return_tensors="pt", add_special_tokens = False)
                len_processed_text = len(processed_tokenized_text["input_ids"][0])  #this could be more than max_tokens without min sentence, which caused fatal error

                if len(processed_text)==0:
                    break

                chunk_list.append(processed_text)  
                instruction_list.append(processed_text)
                #warnings.append(warning)
                
                start_id += len_processed_text # taking from process_tokenized_text to prevent the id from getting wrong

            return chunk_list  #, instruction_list, warnings, split_rule_keys
        

        start = time.time()

        all_chunks = []
        all_file_paths = []
        chunk_dict = {}
        #split_rule_dict = {}
        #warning_chunk_dict = {}

        for file_path in self.file_paths:
            try:
                with open(file_path) as f:
                    text = f.read()
            except:
                continue
            debug = False
            if file_path == "../gkv-code/src/gkvp_geom.f90":
                debug = True
                
            chunks = split_into_chunks(self.tokenizer, text, self.max_tokens, self.min_tokens, self.rules, debug)  #insts, warnings, split_rule_keys
            
            all_chunks += chunks
            fp = [file_path for i in range(len(chunks))]
            all_file_paths += fp
            
            for i in range(len(chunks)):
                chunk_dict[file_path] = chunks
                #if file_path in chunk_dict: chunk_dict[file_path][file_path+str(i)] = chunks[i]
                #else: chunk_dict[file_path] = {file_path+str(i):chunks[i]}
                
                #if warnings[i] == 1: warning_chunk_dict[file_path+str(i)] = chunks[i]

                #split_rule_dict[file_path] = split_rule_keys
            
        end = time.time()

        print("total num chunk: ", len(all_chunks))
        print("process time: ", end - start)


        input_file_path = f"{self.save_path}/chunks.json"
        with open(input_file_path, 'w') as json_file:
            json.dump(all_chunks, json_file)

        file_path_json = f"{self.save_path}/file_paths.json"
        with open(file_path_json, 'w') as json_file:
            json.dump(all_file_paths, json_file)

        file_path_json = f"{self.save_path}/chunk_dict.json"
        with open(file_path_json, 'w') as json_file:
            json.dump(chunk_dict, json_file)
            
        print("file saved")

        # chunks: [<str> chunk of the text, ...]
        # insts: [<str> instructions corresponds to chunk, ...]


    def gather_save_dirs(self, save_dirs, new_save_dir):
        #all_chunk_dict = {}
        all_chunks = []
        all_file_paths = []
        
        for save_dir in save_dirs:
            #with open(f"{save_dir}/chunk_dict.json") as json_file:
            #    chunk_dict = json.load(json_file)
            with open(f"{save_dir}/chunks.json") as json_file:
                chunks = json.load(json_file)
            with open(f"{save_dir}/file_paths.json") as json_file:
                file_paths = json.load(json_file)

            #all_chunk_dict = all_chunk_dict + chunk_dict
            all_chunks += chunks
            all_file_paths += file_paths

        #with open(f"{new_save_dir}/chunk_dict.json", "w") as json_file:
        #    json.dump(all_chunk_dict, json_file)
        with open(f"{new_save_dir}/chunks.json", "w") as json_file:
            json.dump(all_chunks, json_file)
        with open(f"{new_save_dir}/file_paths.json", "w") as json_file:
            json.dump(all_file_paths, json_file)

        print("save_dir updated")
        

    def modify_chunks_manually(self):
        global path_id, chunk_dict, path_id, manually_modified_path
        file_path_json = f"{self.save_path}/chunk_dict.json"
        with open(file_path_json) as json_file:
            chunk_dict = json.load(json_file)

        if not os.path.exists(f"{self.save_path}/manually_modified_path.json"):
            with open(f"{self.save_path}/manually_modified_path.json", "w") as json_file:
                json.dump([], json_file)
            manually_modified_path = []

        else:
            with open(f"{self.save_path}/manually_modified_path.json") as json_file:
                manually_modified_path = json.load(json_file)

        path_id = 0
        chunk_path_list = list(chunk_dict.keys())

        def get_chunk_text():
            global path_id

            while (chunk_path_list[path_id] in manually_modified_path):
                path_id += 1
                if (path_id >= len(chunk_path_list)):
                    return None
                
            chunk_text = ""
            for chunk in chunk_dict[chunk_path_list[path_id]]:
                chunk_text += chunk + "\n---[SPLIT]---\n"
            chunk_text = chunk_text[:-15]
            
            return chunk_text

        app = Flask(__name__)

        @app.route('/')
        def index():
            return render_template('index.html')

        @app.route('/send_data', methods=['POST'])
        def send_data():
            data = get_chunk_text()
            return jsonify({"data": data})

        @app.route('/receive_data', methods=['POST'])
        def receive_data():
            global chunk_dict, path_id, manually_modified_path
            received_data = request.json['data']
            print(f"{chunk_path_list[path_id]} has been modified")

            modified_chunks = received_data.split("---[SPLIT]---") 
            chunk_dict[chunk_path_list[path_id]] = modified_chunks
            with open(f"{self.save_path}/chunk_dict.json", "w") as json_file:
                json.dump(chunk_dict, json_file)

            manually_modified_path.append(chunk_path_list[path_id])
            with open(f"{self.save_path}/manually_modified_path.json", "w") as json_file:
                json.dump(manually_modified_path, json_file)

            print(f"{100 * len(manually_modified_path) / len(chunk_path_list)} % has been finished")
            
            path_id += 1
            data = get_chunk_text()
            
            print(f"{chunk_path_list[path_id]} is being modified")
            
            return jsonify({"status": "success", "new_data": data})
        

        app.run()
        #if __name__ == '__main__':
        #    app.run()



    def finish_modifying(self):

        # load chunk_dict
        file_path_json = f"{self.save_path}/chunk_dict.json"
        with open(file_path_json) as json_file:
            chunk_dict = json.load(json_file)


        all_chunks = []
        all_file_paths = []

        for key in chunk_dict:
            chunk_num = len(chunk_dict[key])
            fp = [key for _ in range(chunk_num)]

            all_file_paths += fp
            all_chunks += chunk_dict[key]
            

        # save others
        input_file_path = f"{self.save_path}/chunks.json"
        with open(input_file_path, 'w') as json_file:
            json.dump(all_chunks, json_file)

        file_path_json = f"{self.save_path}/file_paths.json"
        with open(file_path_json, 'w') as json_file:
            json.dump(all_file_paths, json_file)

        print("file_saved")