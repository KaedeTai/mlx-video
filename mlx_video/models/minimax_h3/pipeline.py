"""H3 pipeline glue: tokenize → embed → PackedLayout → sample → decode.

Port target: comfy_extras/nodes_minimax_h3.py (whole file, 337 LOC), specifically:
  - EmptyMiniMaxH3LatentAV.execute
  - MiniMaxH3ImageToVideo.execute (t2va / fl2va)
  - MiniMaxH3ReferenceToVideo.execute (ref2va)
  - MiniMaxH3SigmaShift.execute

Video/audio latent shape helpers:
  - align_frame_count(n): snap up to 17k+5
  - video_latent_t(n):    2 if n<=5 else ((n-5)//17)*5+2
  - temporal_shape(len):  (frame_count, video_latent_t, round(duration*40))
  - adapt_canvas(w, h):   768 short edge, 768*1344 area cap, 32-round

Reference sizing:
  - image "match":  min(1, sqrt((W*H)/(w*h))) — no upscale
  - image "max":    min(1, REF_IMAGE_SHORT_EDGE / min(w,h)) — 2048 short edge
"""

# TODO: reference from /tmp/h3_recon/ComfyUI/comfy_extras/nodes_minimax_h3.py:1-337


raise NotImplementedError("Phase 7: implement H3 pipeline glue")
