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

        # http request to 'https://civitai.com/api/v1/images'
        # with query parameters: search, count, skip, nsfw

        # make the request
        response = requests.get('https://civitai.com/api/v1/images', params={
            'token': api_key,
            'search': search,
            'limit': limit,
            'page': page,
            'nsfw': nsfw,
        })

        response_json = response.json()
        items_json = response_json['items']
        # for each item get the prompt, civitaiResources into an object { prompt, civitaiResources: { type: 'lora', modelVersionId: number }[] }
        result = []
        for item in items_json:
            prompt = item['meta']['prompt']

            if not search.lower() in prompt.lower():
                continue

            civitaiResources = []
            for resource in item['meta']['civitaiResources']:
                if resource['type'] == 'lora':
                    civitaiResources.append({'type': 'lora', 'modelVersionId': resource['modelVersionId'], 'weight': resource['weight']})
            result.append({'prompt': prompt, 'civitaiResources': civitaiResources})
        
        # get all the modelVersionIds
        modelVersionIds = []
        for item in result:
            for resource in item['civitaiResources']:
                modelVersionIds.append(resource['modelVersionId'])
        modelVersionIds = list(set(modelVersionIds))

        # look up the names using civitai api
        modelVersionNames = {}
        for modelVersionId in modelVersionIds:
            response = requests.get(f'https://civitai.com/api/v1/models/{modelVersionId}', params={'token': api_key})
            response_json = response.json()
            modelVersionNames[modelVersionId] = response_json['name']


        # append the lora tags to each prompt
        for item in result:
            tags = []
            for resource in item['civitaiResources']:
                model_version_id = resource['modelVersionId']
                weight = resource['weight']
                lora_name = modelVersionNames[model_version_id]
                tags.append(f'<lora:{lora_name}:{weight}>')
            item['prompt'] = item['prompt'] + ' '.join(tags) + ' '

        # remove extra lines from each prompt
        for item in result:
            item['prompt'] = item['prompt'].replace('\n', ' ')
        
        # join the prompts
        text_prompts = '\n'.join([item['prompt'] for item in result])

        return (text_prompts,)
    