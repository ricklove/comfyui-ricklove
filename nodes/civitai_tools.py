import requests

class RL_CivitaiTopImagePrompts:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", ),
                "search": ("STRING", ),
                "limit": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "page": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "nsfw": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_prompts",)
    FUNCTION = "run"

    CATEGORY = "ricklove/civitai"

    def run(self, api_key, search='Person', limit=100, page=0, nsfw=False):
        print(f'RL_CivitaiTopImagePrompts/ START')

        # http request to 'https://civitai.com/api/v1/images'
        # with query parameters: search, count, skip, nsfw

        # make the request
        response = requests.get('https://civitai.com/api/v1/images', params={
            'token': api_key,
            'search': search,
            'limit': limit,
            'page': page,
            'nsfw': nsfw,
            'sort': 'Most Reactions',
            'period': 'Week',
            # 'ModelId': 0,
        })

        print(f'RL_CivitaiTopImagePrompts/ response.json()')
        response_json = response.json()
        items_json = response_json['items']
        # for each item get the prompt, civitaiResources into an object { prompt, civitaiResources: { type: 'lora', modelVersionId: number }[] }
        result = []
        for item in items_json:
            print(f'RL_CivitaiTopImagePrompts/(for item in items_json)/ item')

            if not 'meta' in item or item['meta'] is None:
                continue
            if not 'prompt' in item['meta'] or item['meta']['prompt'] is None:
                continue
            prompt = item['meta']['prompt']

            if not search.lower() in prompt.lower():
                continue

            civitaiResources = []
            # get the civitaiResources if it exists on the meta object
            if 'civitaiResources' in item['meta']:
                for resource in item['meta']['civitaiResources']:
                    if 'type' in resource and resource['type'] == 'lora':
                        civitaiResources.append({'type': 'lora', 'modelVersionId': resource['modelVersionId'], 'weight': resource['weight']})
        
            result.append({'id': item['id'], 'prompt': prompt, 'civitaiResources': civitaiResources})
        
        # get all the modelVersionIds
        print(f'RL_CivitaiTopImagePrompts/ get all the modelVersionIds')
        modelVersionIds = []
        for item in result:
            for resource in item['civitaiResources']:
                modelVersionIds.append(resource['modelVersionId'])
        modelVersionIds = list(set(modelVersionIds))

        # look up the names using civitai api
        print(f'RL_CivitaiTopImagePrompts/ look up the names using civitai api')
        modelVersionNames = {}
        for modelVersionId in modelVersionIds:
            print(f'RL_CivitaiTopImagePrompts/ look up the names using civitai api/ modelVersionId: {modelVersionId}')

            response = requests.get(f'https://civitai.com/api/v1/model-versions/{modelVersionId}', params={'token': api_key})
            response_json = response.json()
            if not 'model' in response_json or response_json['model'] is None:
                print(f'RL_CivitaiTopImagePrompts/ look up the names using civitai api modelVersionId: {modelVersionId} model not found')
                continue
            model_json = response_json['model']
            if not 'name' in model_json or model_json['name'] is None:
                print(f'RL_CivitaiTopImagePrompts/ look up the names using civitai api modelVersionId: {modelVersionId} name not found')
                continue
            model_name = model_json['name']
            
            if not 'files' in response_json or response_json['files'] is None:
                print(f'RL_CivitaiTopImagePrompts/ look up the names using civitai api modelVersionId: {modelVersionId} files not found')
                continue
            if len(response_json['files']) == 0:
                print(f'RL_CivitaiTopImagePrompts/ look up the names using civitai api modelVersionId: {modelVersionId} files length is 0')
                continue
            model_file_name = response_json['files'][0]['name']
            modelVersionNames[modelVersionId] = model_file_name


        # append the lora tags to each prompt
        print(f'RL_CivitaiTopImagePrompts/ append the lora tags to each prompt')

        for item in result:
            tags = []
            for resource in item['civitaiResources']:
                model_version_id = resource['modelVersionId']
                weight = resource['weight']
                if not model_version_id in modelVersionNames:
                    print(f'RL_CivitaiTopImagePrompts/ append the lora tags to each prompt/ modelVersionId: {model_version_id} not found for item: {item["id"]}')
                    continue
                lora_name = modelVersionNames[model_version_id]
                tags.append(f'<lora:{lora_name}:{weight}>')
            item['prompt'] = item['prompt'] + ' ' + ' '.join(tags) + ' '

        # remove extra lines from each prompt
        print(f'RL_CivitaiTopImagePrompts/ remove extra lines from each prompt')

        for item in result:
            item['prompt'] = item['prompt'].replace('\n', ' ')
        
        # join the prompts
        print(f'RL_CivitaiTopImagePrompts/ join the prompts')

        text_prompts = '\n'.join([item['prompt'] for item in result])

        return (text_prompts,)
    