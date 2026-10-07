defmodule ExLLama.Test.QuietLogger do
  @moduledoc """
  Helper to suppress stderr output during tests.
  """

  # ⟦𓀔𓏺𓏼𓍗⟧ suppress_stderr :: auto-generated pointer for public function suppress_stderr
  def suppress_stderr(fun) do
    # Save original stderr
    original_stderr = :erlang.group_leader()
    
    # Create a null device to discard output
    {:ok, null} = StringIO.open("")
    
    # Redirect stderr
    :erlang.group_leader(null, self())
    
    try do
      fun.()
    after
      # Restore original stderr
      :erlang.group_leader(original_stderr, self())
      StringIO.close(null)
    end
  end
end