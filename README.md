# abo-dataset-material

Abo material prediction model implementation.
There are no implementation models and weight else.
So, I made it myself following paper.

Rendering uses GGX referenced by [Khronos GLTS2.0 Spec](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#specular-brdf) and [Filament docs](https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf).

# Compelete tasks

1. Open the model and training code
2. open evaluation code
3. Train SVNet and open SVNet weight
4. Upload evaluation result images

# Processing tasks

1. Organize the code
2. MV dataset Class

# Evaluation

Base color, roughness, metallic, rendering loss are measured using RMSE(lower is better),<br>
normal similarity is measured using cosine similarity(higher is better),
Paper column is the metric value written in paper.

| Texture                   | Metric            | Value  | Paper |
| ------------------------- | ----------------- | ------ | ----- |
| Base Color ($\downarrow$) | RMSE              | 0.0543 | 0.129 |
| Roughness ($\downarrow$)  | RMSE              | 0.0677 | 0.063 |
| Metallic ($\downarrow$)   | RMSE              | 0.0739 | 0.170 |
| Normal ($\uparrow$)       | Cosine Similarity | 0.9425 | 0.970 |
| Rendering ($\downarrow$)  | RMSE              | 0.0163 | 0.096 |

# Result Image
![Uploading Untitled.jpeg…]()
