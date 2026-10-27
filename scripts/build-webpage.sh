set -euo pipefail
shopt -s nullglob

PDF_DIR=$1
OUT_DIR=$2

mkdir -p "$OUT_DIR/documents"

for file in $PDF_DIR/*/build/*.pdf
do
  cp "$file" "$OUT_DIR/documents"
done

head_html="<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Notes</title>
    <style>
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            background-color: #ffffff;
            font-family: sans-serif;
        }

        .link-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            gap: 1rem; /* Adjusts the space between newlines */
        }

        .plaintext-link {
            text-decoration: none;
            color: #000000;
        }

        .plaintext-link:hover {
            color: #555555; /* Optional subtle hover effect */
        }
    </style>
</head>
<body>

    <div class=\"link-container\">
"

tail_html="
</div>

</body>
</html>"

# <a href="https://example.com" class="plaintext-link">First Link</a>

echo "$head_html" > "$OUT_DIR/index.html"

for file in "$OUT_DIR"/documents/*.pdf
do
    rel_path="documents/$(basename "$file")"
    title=$(basename "$file" .pdf)
    echo "<a href=\"$rel_path\" class=\"plaintext-link\">$title</a>" >> "$OUT_DIR/index.html"
done

echo "$tail_html" >> "$OUT_DIR/index.html"
