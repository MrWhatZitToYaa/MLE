import events as event

def reward_from_events(self, event_sequence) -> int:
    """
    TODO: Datenobjekt das die events in sequence speichert erstellen und hier als parameter statt event benutzen
    Returns the total reward from a series of events
    """
    rewards = {
        event.MOVED_LEFT: 5,
        event.MOVED_RIGHT: 5,
        event.MOVED_UP: 5,
        event.MOVED_DOWN: 5,
        event.WAITED: -5,
        event.INVALID_ACTION: -10,
        
        event.BOMB_DROPPED: 0,
        event.BOMB_EXPLODED: 0,
        
        event.CRATE_DESTROYED: 5,
        event.COIN_FOUND: 1,
		event.COIN_COLLECTED: 10,
        
        event.KILLED_OPPONENT: 0,
        event.KILLED_SELF: -100,
        
		event.GOT_KILLED: -50,
        event.OPPONENT_ELIMINATED: -2,
		event.SURVIVED_ROUND: 50,
    }
    
    total_reward = 0
    for instance in event_sequence:
        if instance in rewards:
            total_reward += rewards[instance]
            
    self.logger.info(f"Gained {total_reward} total reward for events {', '.join(event_sequence)}")
    return total_reward
