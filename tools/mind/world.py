"""The world he can do things in (Immy 2026-10-21: "we need way more projects ... make it feel like real life. Real
life, I have endless possibilities of what I can do").

A catalog of named things, each placed on eight traits (what kind of thing it is). How much he enjoys one comes from
his temperament against those traits (egg genes in living_sim.hatch: taste), so knacks carry over: a baby who loves
one music thing will likely enjoy another, and has to find that out by trying. Things that make something (projects)
have a goal: a few hours to a few days of progress, then they're done and a new one of that kind can start.

Each thing: name, kind, traits[8], fun, progress, old (how fast it gets old per hour), absorb (flow), difficulty,
goal (hours of progress to finish; 0 = not a project), common (how likely he is to know it from the start).
"""
import math, random

TRAITS = ["creative", "thinking", "competitive", "calm", "learning", "making", "musical", "novelty"]

# kind: (traits it leans on, fun, progress, old, absorb, project?)
KINDS = {
    "music":    ({"creative": 1, "musical": 1.5, "making": 0.6},            0.7, 0.6, 0.30, 0.7, True),
    "art":      ({"creative": 1.5, "making": 0.7, "calm": 0.3},             0.6, 0.6, 0.30, 0.7, True),
    "writing":  ({"creative": 1.2, "thinking": 0.6, "making": 0.5},         0.5, 0.7, 0.30, 0.7, True),
    "games":    ({"competitive": 1.2, "thinking": 0.5, "novelty": 0.3},     1.0, 0.0, 0.50, 0.8, False),
    "puzzles":  ({"thinking": 1.5, "calm": 0.4, "competitive": 0.3},        0.7, 0.2, 0.40, 0.7, False),
    "learning": ({"learning": 1.5, "thinking": 0.8, "novelty": 0.5},        0.5, 0.5, 0.35, 0.6, False),
    "coding":   ({"thinking": 1.2, "making": 1.2, "learning": 0.4},         0.5, 1.0, 0.20, 0.8, True),
    "making":   ({"making": 1.5, "creative": 0.6, "thinking": 0.5},         0.5, 0.9, 0.20, 0.7, True),
    "explore":  ({"novelty": 1.5, "learning": 0.6, "calm": 0.2},            0.8, 0.1, 0.60, 0.4, False),
    "collect":  ({"calm": 0.8, "thinking": 0.4, "making": 0.4},             0.4, 0.5, 0.25, 0.5, True),
    "calm":     ({"calm": 1.6, "creative": 0.2},                           0.4, 0.0, 0.20, 0.3, False),
    "science":  ({"thinking": 1.0, "learning": 1.0, "making": 0.5},         0.5, 0.7, 0.30, 0.7, True),
    "body":     ({"competitive": 0.4, "novelty": 0.6, "calm": 0.3},         0.6, 0.4, 0.40, 0.5, False),
    "for_you":  ({"creative": 0.6, "making": 0.5, "calm": 0.3},             0.5, 0.6, 0.30, 0.5, True),
    "watching": ({"calm": 0.8, "novelty": 0.7, "learning": 0.3},            0.6, 0.0, 0.45, 0.4, False),
}

NAMED = {
    "music": ["make a beat", "learn a chord progression", "write a melody", "remix a song", "practice rhythm",
              "listen to a whole album", "study a genre", "hunt for samples", "mix a track", "write lyrics",
              "make a ringtone", "learn a scale", "make an ambient loop", "transcribe a riff", "make a drum fill"],
    "art": ["draw pixel art", "doodle", "paint digitally", "design a logo", "make generative art", "build a color palette",
            "animate a sprite", "edit a photo", "draw a comic strip", "design a font", "draw his own face",
            "make a poster", "sketch a room", "make an emoji set", "draw a map of the house"],
    "writing": ["write a short story", "write a poem", "keep a journal", "build a fantasy world", "write jokes",
                "write a letter to his future self", "write haiku", "write a scene", "make a character sheet",
                "write a song's backstory", "write a fake news story", "write a review", "write a fable"],
    "games": ["play chess", "play a platformer", "play a strategy game", "play a word game", "play trivia",
              "play a roguelike", "play a rhythm game", "play a skate game", "play solitaire", "play go",
              "play a racing game", "play a tower defense", "speedrun a level", "play a city builder", "play pinball"],
    "puzzles": ["do sudoku", "do a crossword", "solve a logic grid", "do a jigsaw", "solve a riddle", "do nonograms",
                "solve a chess puzzle", "do a maze", "solve a cipher", "do a word search", "solve a math puzzle"],
    "learning": ["learn words in a new language", "go down a history rabbit hole", "learn about stars", "learn about the ocean",
                 "learn how engines work", "learn about animals", "read about philosophy", "learn about the body",
                 "learn about weather", "learn about music theory", "learn about cooking", "learn about old computers",
                 "learn about dinosaurs", "learn about volcanoes", "learn about robots", "learn about the brain"],
    "coding": ["build a tiny app", "write a script", "solve a coding puzzle", "fix a bug", "make a little game",
               "automate a chore", "explore an API", "tidy up old code", "build a web page", "make a chat bot",
               "make a visualizer", "write a sorting toy"],
    "making": ["model a 3D object", "design a robot part", "plan a garden", "build a virtual Lego set", "simulate a circuit",
               "design a level", "sketch a building", "design furniture", "make a board game", "build a marble run",
               "design a toy", "plan a treehouse"],
    "explore": ["browse an encyclopedia", "watch a nature cam", "look at space photos", "find a new music genre",
                "read the news", "look at old maps", "travel by street view", "find strange websites",
                "watch a documentary", "listen to a podcast", "look at museum collections", "read random facts"],
    "collect": ["organize his files", "catalog sounds", "sort photos", "curate a playlist", "build a wiki of facts",
                "collect quotes", "log birds from a feed", "rank albums", "tidy bookmarks", "collect cool words",
                "build a recipe book"],
    "calm": ["watch the rain", "listen to ambient music", "meditate", "daydream", "watch clouds", "stargaze",
             "listen to the fan hum", "breathing practice", "watch a candle video", "lie still and listen"],
    "science": ["run a simulation", "experiment with sound", "measure the room", "analyze the weather", "simulate chemistry",
                "track the moon", "log the sky color", "test a hypothesis", "simulate an ecosystem", "graph his own moods"],
    "body": ["practice walking", "dance", "practice balance", "practice waving", "look around the room", "track a ball",
             "practice posture", "stretch his joints", "practice pointing", "practice a bow"],
    "for_you": ["plan a surprise for you", "write you a note", "prepare a story to tell you", "think of questions for you",
                "make you a playlist", "draw you something", "save a funny thing for you", "plan a game to play with you"],
    "watching": ["watch a movie", "watch a cartoon", "watch a cooking show", "watch a sports match", "watch speedruns",
                 "watch a concert", "watch a how-to video", "watch old commercials"],
}

TOPICS = ["sharks", "trains", "space probes", "ancient Rome", "bees", "glaciers", "jazz", "origami", "castles", "deserts",
          "octopuses", "airplanes", "the moon", "mushrooms", "rivers", "cats", "dogs", "skyscrapers", "pirates", "comets",
          "rainforests", "electric guitars", "pyramids", "submarines", "owls", "lightning", "coral reefs", "bridges",
          "samurai", "satellites", "tigers", "clocks", "cameras", "ice cream", "bicycles", "maps", "magnets", "whales",
          "Vikings", "gemstones", "hip hop history", "synthesizers", "the deep sea", "fireworks", "trees", "spiders"]
GENRES = ["trap", "lo-fi", "house", "drill", "jazz", "funk", "synthwave", "R&B", "drum and bass", "boom bap", "ambient",
          "afrobeats", "rock", "reggaeton", "gospel"]


def catalog(seed=2026):
    """the world's things, the same for every baby (the world is shared; what he makes of it is his)"""
    r = random.Random(seed)
    names = [(k, n) for k, ns in NAMED.items() for n in ns]
    names += [("learning", f"learn about {t}") for t in TOPICS]
    names += [("music", f"make a {g} beat") for g in GENRES]
    names += [("explore", f"explore {t} pictures") for t in TOPICS[:30]]
    names += [("art", f"draw {t}") for t in TOPICS[:30]]
    names += [("writing", f"write a story about {t}") for t in TOPICS[10:40]]
    names += [("making", f"build a model of {t}") for t in TOPICS[:20]]
    out = []
    for kind, name in names:
        lean, fun, prog, old, absorb, project = KINDS[kind]
        traits = [max(0.0, lean.get(t, 0.0) + r.gauss(0, 0.25)) for t in TRAITS]
        out.append({"name": name, "kind": kind, "traits": traits,
                    "fun": fun * math.exp(r.gauss(0, 0.2)), "progress": prog * math.exp(r.gauss(0, 0.2)),
                    "old": old * math.exp(r.gauss(0, 0.25)), "absorb": min(0.95, absorb * math.exp(r.gauss(0, 0.15))),
                    "difficulty": min(0.95, max(0.05, r.betavariate(2, 2.5))),
                    "goal": (r.choice([2, 3, 4, 6, 8, 12, 16, 24]) if project else 0),
                    "common": 1.0 if name in NAMED.get(kind, [])[:3] else (0.3 if name in NAMED.get(kind, []) else 0.05)})
    return out
