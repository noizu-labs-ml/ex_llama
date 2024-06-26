defmodule ExLLama.ChatTemplate.AmberChat do
  @moduledoc """
  based on: [https://github.com/chujiezheng/chat_templates/blob/main/chat_templates/amberchat.jinja]
  ```jinja
  {% if messages[0]['role'] == 'system' %}
    {% set loop_messages = messages[1:] %}
    {% set system_message = messages[0]['content'].strip() + '\n' %}
  {% else %}
    {% set loop_messages = messages %}
    {% set system_message = '' %}
  {% endif %}

  {% for message in loop_messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}

    {% if loop.index0 == 0 %}
        {{ bos_token + system_message }}
    {% endif %}

    {% if message['role'] == 'user' %}
        {{ '###Human: ' + message['content'].strip() + '\n' }}
    {% elif message['role'] == 'assistant' %}
        {{ '###Assistant: ' + message['content'].strip() + '\n' }}
    {% endif %}

    {% if loop.last and message['role'] == 'user' and add_generation_prompt %}
        {{ '###Assistant:' }}
    {% endif %}
  {% endfor %}
  ````
  """
end
