package ltc_fit

// Fits the LTC table for the rasterizer's area lights and writes
// realtime/ltc/ltc_table.bin. See realtime/ltc/ltc.odin for what is fitted.
//
//   odin run tools/ltc_fit -o:speed
//
// Deterministic: stratified sampling and a fixed walk order, so a rerun on the
// same BRDF writes the same bytes. Rerun whenever the BRDF in shading.slang
// (and so in ltc.odin) changes.

import "core:fmt"
import "core:os"
import "core:slice"
import "core:sync"
import "core:thread"
import "core:time"

import ltc "../../realtime/ltc"

OUTPUT :: "realtime/ltc/" + ltc.TABLE_FILE

table: ltc.Table
first_row: [ltc.SIZE][3]f32
next_column: int
done_columns: int

main :: proc() {
	start := time.tick_now()

	// Normal incidence, roughest to smoothest: each column's starting guess.
	guess := [3]f32{1, 1, 0}
	for x := ltc.SIZE - 1; x >= 0; x -= 1 {
		params, entry := ltc.fit_cell(x, 0, guess)
		table[x] = entry
		first_row[x] = params
		guess = params
	}

	// Then every column toward grazing, in parallel.
	workers := max(os.get_processor_core_count() - 1, 1)
	threads := make([]^thread.Thread, workers)
	defer delete(threads)
	for &t in threads {
		t = thread.create_and_start(proc() {
			for {
				x := sync.atomic_add(&next_column, 1)
				if x >= ltc.SIZE { return }
				guess := first_row[x]
				for y in 1 ..< ltc.SIZE {
					params, entry := ltc.fit_cell(x, y, guess)
					table[y * ltc.SIZE + x] = entry
					guess = params
				}
				n := sync.atomic_add(&done_columns, 1) + 1
				fmt.printf("\r%d / %d columns", n, ltc.SIZE)
			}
		})
	}
	thread.join_multiple(..threads)
	for t in threads { thread.destroy(t) }

	// The horizon table, which shares the file; see Entry.sphere.
	for y in 0 ..< ltc.SIZE {
		for x in 0 ..< ltc.SIZE {
			table[y * ltc.SIZE + x].sphere = ltc.sphere_cell(x, y)
		}
	}

	if err := os.write_entire_file(OUTPUT, slice.to_bytes(table[:])); err != nil {
		fmt.eprintln("\nwriting", OUTPUT, "failed:", err)
		os.exit(1)
	}
	fmt.printfln("\nwrote %s in %.1f s", OUTPUT, time.duration_seconds(time.tick_since(start)))
}
