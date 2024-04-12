defmodule ExLLama.ChatTemplate.Alpaca do
  @moduledoc """
  based on: [https://github.com/chujiezheng/chat_templates/blob/main/chat_templates/alpaca.jinja]
  ```jinja
  {% if messages[0]['role'] == 'system' %}
    {% set loop_messages = messages[1:] %}
    {% set system_message = messages[0]['content'].strip() + '\n\n' %}
  {% else %}
    {% set loop_messages = messages %}
    {% set system_message = '' %}
  {% endif %}

  {{ bos_token + system_message }}
  {% for message in loop_messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}

    {% if message['role'] == 'user' %}
        {{ '### Instruction:\n' + message['content'].strip() + '\n\n' }}
    {% elif message['role'] == 'assistant' %}
        {{ '### Response:\n' + message['content'].strip() + eos_token + '\n\n' }}
    {% endif %}

    {% if loop.last and message['role'] == 'user' and add_generation_prompt %}
        {{ '### Instruction:\n' }}
    {% endif %}
  {% endfor %}
  ````
  """
end
