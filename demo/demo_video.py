"""
Demo for doing interesting things with a video
"""
import sys
sys.path.append('../')

from mreserve.preprocess import video_to_segments, preprocess_video, encoder, MASK
from mreserve.modeling import PretrainedMerlotReserve
import jax
import jax.numpy as jnp

# This handles loading the model and getting the checkpoints.
grid_size = (18, 32)
model = PretrainedMerlotReserve.from_pretrained(model_name='large', image_grid_size=grid_size)

## First open the video and break it up into segments. you can only have 8.
# Each segment is 5 seconds so it corresponds to seconds 15 - 55 of the video

# Feel free to change the URL!
video_segments = video_to_segments('pmjPjZZRhNQ.mp4')
video_segments = video_segments[3:11]

# Set up a fake classification task.
video_segments[0]['text'] = 'in this video i\'ll be<|MASK|>'
video_segments[0]['use_text_as_input'] = True
for i in range(1,8):
    video_segments[i]['use_text_as_input'] = False

video_pre = preprocess_video(video_segments, output_grid_size=grid_size, verbose=True)

# Now we embed the entire video and extract the text. result is  [seq_len, H]. we extract a hidden state for every
# MASK token
out_h = model.embed_video(**video_pre)
out_h = out_h[video_pre['tokens'] == MASK]

options = ['making coffee', 'going backpacking']

# the following is all the labels from activitynet. why not! some of them don't make sense grammatically though.
options += ['Applying sunscreen', 'Archery', 'Arm wrestling', 'Assembling bicycle', 'BMX', 'Baking cookies', 'Ballet', 'Bathing dog', 'Baton twirling', 'Beach soccer', 'Beer pong', 'Belly dance', 'Blow-drying hair', 'Blowing leaves', 'Braiding hair', 'Breakdancing', 'Brushing hair', 'Brushing teeth', 'Building sandcastles', 'Bullfighting', 'Bungee jumping', 'Calf roping', 'Camel ride', 'Canoeing', 'Capoeira', 'Carving jack-o-lanterns', 'Changing car wheel', 'Cheerleading', 'Chopping wood', 'Clean and jerk', 'Cleaning shoes', 'Cleaning sink', 'Cleaning windows', 'Clipping cat claws', 'Cricket', 'Croquet', 'Cumbia', 'Curling', 'Cutting the grass', 'Decorating the Christmas tree', 'Disc dog', 'Discus throw', 'Dodgeball', 'Doing a powerbomb', 'Doing crunches', 'Doing fencing', 'Doing karate', 'Doing kickboxing', 'Doing motocross', 'Doing nails', 'Doing step aerobics', 'Drinking beer', 'Drinking coffee', 'Drum corps', 'Elliptical trainer', 'Fixing bicycle', 'Fixing the roof', 'Fun sliding down', 'Futsal', 'Gargling mouthwash', 'Getting a haircut', 'Getting a piercing', 'Getting a tattoo', 'Grooming dog', 'Grooming horse', 'Hammer throw', 'Hand car wash', 'Hand washing clothes', 'Hanging wallpaper', 'Having an ice cream', 'High jump', 'Hitting a pinata', 'Hopscotch', 'Horseback riding', 'Hula hoop', 'Hurling', 'Ice fishing', 'Installing carpet', 'Ironing clothes', 'Javelin throw', 'Kayaking', 'Kite flying', 'Kneeling', 'Knitting', 'Laying tile', 'Layup drill in basketball', 'Long jump', 'Longboarding', 'Making a cake', 'Making a lemonade', 'Making a sandwich', 'Making an omelette', 'Mixing drinks', 'Mooping floor', 'Mowing the lawn', 'Paintball', 'Painting', 'Painting fence', 'Painting furniture', 'Peeling potatoes', 'Ping-pong', 'Plastering', 'Plataform diving', 'Playing accordion', 'Playing badminton', 'Playing bagpipes', 'Playing beach volleyball', 'Playing blackjack', 'Playing congas', 'Playing drums', 'Playing field hockey', 'Playing flauta', 'Playing guitarra', 'Playing harmonica', 'Playing ice hockey', 'Playing kickball', 'Playing lacrosse', 'Playing piano', 'Playing polo', 'Playing pool', 'Playing racquetball', 'Playing rubik cube', 'Playing saxophone', 'Playing squash', 'Playing ten pins', 'Playing violin', 'Playing water polo', 'Pole vault', 'Polishing forniture', 'Polishing shoes', 'Powerbocking', 'Preparing pasta', 'Preparing salad', 'Putting in contact lenses', 'Putting on makeup', 'Putting on shoes', 'Rafting', 'Raking leaves', 'Removing curlers', 'Removing ice from car', 'Riding bumper cars', 'River tubing', 'Rock climbing', 'Rock-paper-scissors', 'Rollerblading', 'Roof shingle removal', 'Rope skipping', 'Running a marathon', 'Sailing', 'Scuba diving', 'Sharpening knives', 'Shaving', 'Shaving legs', 'Shot put', 'Shoveling snow', 'Shuffleboard', 'Skateboarding', 'Skiing', 'Slacklining', 'Smoking a cigarette', 'Smoking hookah', 'Snatch', 'Snow tubing', 'Snowboarding', 'Spinning', 'Spread mulch', 'Springboard diving', 'Starting a campfire', 'Sumo', 'Surfing', 'Swimming', 'Swinging at the playground', 'Table soccer', 'Tai chi', 'Tango', 'Tennis serve with ball bouncing', 'Throwing darts', 'Trimming branches or hedges', 'Triple jump', 'Tug of war', 'Tumbling', 'Using parallel bars', 'Using the balance beam', 'Using the monkey bar', 'Using the pommel horse', 'Using the rowing machine', 'Using uneven bars', 'Vacuuming floor', 'Volleyball', 'Wakeboarding', 'Walking the dog', 'Washing dishes', 'Washing face', 'Washing hands', 'Waterskiing', 'Waxing skis', 'Welding', 'Windsurfing', 'Wrapping presents', 'Zumba']
label_space = model.get_label_space(options)

# Dot product the <|MASK|> tokens and the options together
logits = 100.0 * jnp.einsum('bh,lh->bl', out_h, label_space)

for i, logits_i in enumerate(logits):
    print(f"Idx {i}", flush=True)
    probs = jax.nn.softmax(logits_i, -1)
    for idx_i in jnp.argsort(-probs):
        p_i = probs[idx_i]
        print("{:.1f} {}".format(p_i * 100.0, options[idx_i], flush=True))
