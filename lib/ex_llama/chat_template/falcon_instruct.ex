defmodule ExLLama.ChatTemplate.FalconInstruct do
  @moduledoc """
  based on: [https://github.com/chujiezheng/chat_templates/blob/main/chat_templates/falcon-instruct.jinja]]
  ```jinja
  {% if messages[0]['role'] == 'system' %}
    {% set loop_messages = messages[1:] %}
    {% set system_message = messages[0]['content'] %}
  {% else %}
    {% set loop_messages = messages %}
    {% set system_message = '' %}
  {% endif %}

  {% for message in loop_messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}

    {% if loop.index0 == 0 %}
        {{ system_message.strip() }}
    {% endif %}
    {{ '\n\n' + message['role'].title() + ': ' + message['content'].strip().replace('\r\n', '\n').replace('\n\n', '\n') }}

    {% if loop.last and message['role'] == 'user' and add_generation_prompt %}
        {{ '\n\nAssistant:' }}
    {% endif %}
  {% endfor %}
  ````
  """
end
