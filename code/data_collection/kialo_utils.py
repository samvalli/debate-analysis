import json, re

def kialo_parser(input_dir,output_dir):
    
    question_flag=0
    with open(input_dir, 'r') as fi:
        lines = []
        line_text=''
        for line in fi:
            if line=='\n':
                lines.append(line_text)
                line_text = ''
            else:
                line_text+=line.replace('\n','')

        result = []

        lines.pop(0)
        if 'Question' in lines[0]:
            lines.pop(0)
            question_flag=1
        
        for line in lines:
            if line.lstrip().startswith('Sources:'):
                break
            # find the tree position the comment is in
            tree =  re.search(r"^\s*(\d{1,}.)+", line)
            parsed = re.findall(r"(\d{1,}(?=\.))+", tree.group())
            if question_flag==0:
                level = len(parsed)-1
            if question_flag==1:
                level = len(parsed)-2
            # find if the comment is Pro or Con
            if level==0:
                stance='Thesis'
            else:
                stance = re.search(r"(Con|Pro)(?::)", line)
                stance=stance.group(1)
            # find the text of the comment
            if level==0:
                content = re.search(r"Thesis:\s*(.+?)(\[\d+\])?$", line)
                content = content.group(1).strip()
            else:    
                content = re.search(r"((Con|Pro)(?::\s))(.*)", line)
                content = content.group(3)
            # define the hierarchy of the current comment
            # which is based on the tree structure
            
            # make a dictionary with the single entry
            # and put it at the end of the list
            result.append({
                "Tree": tree.group(),
                "Level": level,
                "Stance": stance,
                "Content": content
            })

        to_write = json.dumps(result, sort_keys=True, indent=4, separators=(',', ': '))

    with open(output_dir, 'w') as fo:
        # print to_write
        fo.write(to_write)
        output_message = "Operation completed. Wrote " + str(len(result)) + " entries in " + str(output_dir)
        print("=" * len(output_message))
        print(output_message)
        print("=" * len(output_message))


def count_kialo_items(input_dir):
    
    question_flag=0
    with open(input_dir, 'r') as fi:
        lines = []
        line_text=''
        for line in fi:
            if line=='\n':
                lines.append(line_text)
                line_text = ''
            else:
                line_text+=line.replace('\n','')

        # list containing each parsed comment
        result = []


        # we remove the first two lines of the text
        # as we don't need the header
        lines.pop(0)
        if 'Question' in lines[0]:
            lines.pop(0)
            question_flag=1

        ##                                            ##
        ##                 REGEDITS                   ##
        ##                                            ##
        # iterate every row in the text file
        
        for line in lines:
            if line.lstrip().startswith('Sources:'):
                break
            # find the tree position the comment is in
            tree =  re.search(r"^\s*(\d{1,}.)+", line)
            parsed = re.findall(r"(\d{1,}(?=\.))+", tree.group())
            if question_flag==0:
                level = len(parsed)-1
            if question_flag==1:
                level = len(parsed)-2
            # find if the comment is Pro or Con
            if level==0:
                stance='Thesis'
            else:
                stance = re.search(r"(Con|Pro)(?::)", line)
                stance=stance.group(1)
            # find the text of the comment
            if level==0:
                content = re.search(r"Thesis:\s*(.+?)(\[\d+\])?$", line)
                content = content.group(1).strip()
            else:    
                content = re.search(r"((Con|Pro)(?::\s))(.*)", line)
                content = content.group(3)
            # define the hierarchy of the current comment
            # which is based on the tree structure
            
            # make a dictionary with the single entry
            # and put it at the end of the list
            result.append({
                "Tree": tree.group(),
                "Level": level,
                "Stance": stance,
                "Content": content
            })

    return len(result)

def extract_participants(file_path):
    authors = []
    edits = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Match the "Profile picture for [name]" pattern
        match = re.match(r'Profile picture for (.+)', line)
        if match:
            name = match.group(1)
            # The next line should be the username
            i += 1
            username = lines[i].strip()
            # The line after that should contain the numbers
            i += 1
            numbers = list(map(int, lines[i].strip().split('\t')))
            if (numbers[0] + numbers[1])>0:
                authors.append(username)
                edits.append(numbers[0]+numbers[1])
        i += 1
    return authors,edits

