from docling_core.types.doc import DocItemLabel


def get_page_numbers(item):
    return sorted({
        prov.page_no
        for prov in (item.prov or [])
    })


def create_section(item):
    return {
        "heading": item.text,
        "level": item.level,
        "page_numbers": get_page_numbers(item),
        "content": [],
        "children": [],
    }


def extract_document_structure(doc):

    root = {
        "heading": None,
        "level": 0,
        "content": [],
        "children": [],
    }

    stack = [root]

    for item, _ in doc.iterate_items():

        # A. Process headings
        if item.label == DocItemLabel.SECTION_HEADER:

            section = create_section(item)

            # Find the correct parent section
            while (
                len(stack) > 1
                and stack[-1]["level"] >= section["level"]
            ):
                stack.pop()

            # Attach the new section
            stack[-1]["children"].append(section)

            # Make it the active section
            stack.append(section)

        # B. Process text-bearing items
        elif getattr(item, "text", None):
            text = item.text.strip()

            if not text:
                continue

            label = item.label.value

            content_type = {
                "text": "paragraph",
                "list_item": "list_item",
                "caption": "caption",
                "footnote": "footnote",
                "code": "code",
                "formula": "formula",
            }.get(label, label)

            content = {
                "type": content_type,
                "text": text,
                "page_numbers": get_page_numbers(item),
            }

            stack[-1]["content"].append(content)

    return root