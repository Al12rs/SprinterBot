from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.gamestates import GameState


class TouchBallTerminalCondition(TerminalCondition):
  def reset(self, initial_state: GameState):
    pass

  def is_terminal(self, current_state: GameState) -> bool:
    return current_state.last_touch != -1